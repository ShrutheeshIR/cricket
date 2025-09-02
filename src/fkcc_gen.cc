#include "pinocchio_cppadcg.hh"

#include <pinocchio/parsers/urdf.hpp>
#include <pinocchio/parsers/srdf.hpp>
#include <pinocchio/algorithm/joint-configuration.hpp>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/algorithm/geometry.hpp>
#include <pinocchio/multibody/geometry.hpp>
#include <pinocchio/collision/collision.hpp>

#include <coal/shape/geometric_shapes.h>

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Min_sphere_of_spheres_d.h>
#include <CGAL/Min_sphere_of_spheres_d_traits_3.h>

#include <fmt/core.h>
#include <nlohmann/json.hpp>
#include <inja/inja.hpp>
#include <cxxopts.hpp>

#include <filesystem>
#include <stdexcept>
#include <vector>
#include <optional>

#include "lang_gen.hh"

using namespace pinocchio;
using namespace CppAD;
using namespace CppAD::cg;

// Typedef for AD types
using CGD = CG<double>;
using ADCG = AD<CGD>;

using ADModel = ModelTpl<ADCG>;
using ADData = DataTpl<ADCG>;
using ADVectorXs = Eigen::Matrix<ADCG, Eigen::Dynamic, 1>;
using ADMatrixXs = Eigen::Matrix<ADCG, Eigen::Dynamic, Eigen::Dynamic>;

struct SphereInfo
{
    std::size_t geom_index;
    float radius;
    std::size_t parent_joint;
    std::size_t parent_frame;
    SE3 relative;
};

auto min_sphere_of_spheres(const std::vector<SphereInfo> &info) -> std::array<float, 4>
{
    using K = CGAL::Exact_predicates_inexact_constructions_kernel;
    using Traits = CGAL::Min_sphere_of_spheres_d_traits_3<K, double>;
    using Sphere = Traits::Sphere;
    using Point = K::Point_3;
    using MinSphere = CGAL::Min_sphere_of_spheres_d<Traits>;

    std::vector<Sphere> cgal_spheres;
    cgal_spheres.reserve(info.size());

    for (const auto &sphere : info)
    {
        auto pos = sphere.relative.translation();
        cgal_spheres.emplace_back(Point(pos[0], pos[1], pos[2]), sphere.radius);
    }

    MinSphere ms(cgal_spheres.begin(), cgal_spheres.end());
    std::array<float, 4> sphere;
    std::copy(ms.center_cartesian_begin(), ms.center_cartesian_end(), sphere.begin());
    sphere[3] = ms.radius();
    return sphere;
}

struct RobotInfo
{
    RobotInfo(
        const std::filesystem::path &urdf_file,
        const std::optional<std::filesystem::path> &srdf_file,
        const std::optional<std::string> &end_effector)
    {
        if (not std::filesystem::exists(urdf_file))
        {
            throw std::runtime_error(fmt::format("URDF file {} does not exist!", urdf_file.string()));
        }

        pinocchio::urdf::buildModel(urdf_file, model);
        pinocchio::urdf::buildGeom(model, urdf_file, COLLISION, collision_model);

        if (srdf_file and not std::filesystem::exists(*srdf_file))
        {
            throw std::runtime_error(fmt::format("SRDF file () does not exist!", srdf_file->string()));
        }
        else if (not srdf_file)
        {
            fmt::print("No SRDF file provided, guessing collisions!\n");
            guess_self_collisions();
        }
        else
        {
            collision_model.addAllCollisionPairs();
            pinocchio::srdf::removeCollisionPairs(model, collision_model, *srdf_file);
            extract_collision_data();
        }

        extract_spheres();

        if (not end_effector)
        {
            end_effector_name = model.frames[model.nframes - 1].name;
            fmt::print("No EE provided, using distal link `{}`.\n", end_effector_name);
        }
        else if (not model.existFrame(*end_effector))
        {
            throw std::runtime_error(fmt::format("Invalid EE name {}", *end_effector));
        }
        else
        {
            end_effector_name = *end_effector;
        }

        end_effector_index = model.getFrameId(end_effector_name);
    }

    auto json() -> nlohmann::json
    {
        const Eigen::VectorXd lower_bound = model.lowerPositionLimit;
        const Eigen::VectorXd upper_bound = model.upperPositionLimit;
        const Eigen::VectorXd bound_range = upper_bound - lower_bound;
        const Eigen::VectorXd bound_descale = bound_range.cwiseInverse();

        nlohmann::json json;
        json["n_q"] = model.nq;
        json["n_spheres"] = spheres.size();
        json["bound_lower"] = std::vector<float>(lower_bound.data(), lower_bound.data() + model.nq);
        json["bound_range"] = std::vector<float>(bound_range.data(), bound_range.data() + model.nq);
        json["bound_descale"] = std::vector<float>(bound_descale.data(), bound_descale.data() + model.nq);
        json["measure"] = bound_range.prod();
        json["end_effector"] = end_effector_name;
        json["end_effector_index"] = end_effector_index;
        json["min_radius"] = min_radius;
        json["max_radius"] = max_radius;
        json["joint_names"] = dof_to_joint_names();
        json["allowed_link_pairs"] = allowed_link_pairs;
        json["per_link_spheres"] = per_link_spheres;
        json["links_with_geometry"] = links_with_geometry;
        json["bounding_sphere_index"] = bounding_sphere_index;
        json["end_effector_collisions"] = get_frames_colliding_end_effector();

        std::vector<std::string> link_names;
        for (auto i = 0U; i < model.frames.size(); ++i)
        {
            link_names.emplace_back(model.frames[i].name);
        }
        json["link_names"] = link_names;

        return json;
    }

    auto dof_to_joint_names() -> std::vector<std::string>
    {
        std::vector<std::size_t> dof_to_joint_id(model.nq);
        for (auto joint_id = 1U; joint_id < model.joints.size(); ++joint_id)
        {
            const auto &joint = model.joints[joint_id];
            auto start_idx = joint.idx_q();
            auto nq = joint.nq();

            for (auto i = 0U; i < nq; ++i)
            {
                dof_to_joint_id[start_idx + i] = joint_id;
            }
        }

        std::vector<std::string> dof_to_joint_name(model.nq);
        for (auto i = 0U; i < model.nq; ++i)
        {
            dof_to_joint_name[i] = model.names[dof_to_joint_id[i]];
        }

        return dof_to_joint_name;
    }

    auto get_frames_colliding_end_effector() -> std::vector<std::size_t>
    {
        std::size_t end_effector_joint = model.frames[end_effector_index].parentJoint;

        std::vector<std::size_t> frames;
        for (auto i = 0U; i < model.frames.size(); ++i)
        {
            if (model.frames[i].parentJoint == end_effector_joint)
            {
                if (bounding_spheres.find(i) != bounding_spheres.end())
                {
                    frames.emplace_back(i);
                }
            }
        }

        std::set<std::size_t> end_effector_allowed_collisions;
        for (const auto &[first, second] : allowed_link_pairs)
        {
            if (std::find(frames.begin(), frames.end(), first) != frames.end())
            {
                end_effector_allowed_collisions.emplace(second);
            }

            if (std::find(frames.begin(), frames.end(), second) != frames.end())
            {
                end_effector_allowed_collisions.emplace(first);
            }
        }

        return std::vector<std::size_t>(
            end_effector_allowed_collisions.begin(), end_effector_allowed_collisions.end());
    }

    auto extract_spheres() -> void
    {
        for (auto i = 0U; i < collision_model.ngeoms; ++i)
        {
            const auto &geom_obj = collision_model.geometryObjects[i];
            const auto &sphere_ptr = std::dynamic_pointer_cast<coal::Sphere>(geom_obj.geometry);

            if (sphere_ptr)
            {
                SphereInfo info;
                info.geom_index = i;
                info.radius = sphere_ptr->radius;
                info.parent_joint = geom_obj.parentJoint;
                info.parent_frame = geom_obj.parentFrame;
                info.relative = geom_obj.placement;

                spheres.emplace_back(info);

                min_radius = std::min(min_radius, info.radius);
                max_radius = std::max(max_radius, info.radius);
            }
            else
            {
                throw std::runtime_error(
                    fmt::format("Invalid non-sphere geometry in URDF {}", geom_obj.name));
            }
        }

        std::size_t bs = 0;
        for (auto i = 0U; i < model.frames.size(); ++i)
        {
            std::vector<SphereInfo> link_info;
            std::vector<std::size_t> sphere_indices;
            for (const auto &info : spheres)
            {
                if (info.parent_frame == i)
                {
                    link_info.emplace_back(info);
                    sphere_indices.emplace_back(info.geom_index);
                }
            }

            per_link_spheres.emplace_back(sphere_indices);

            if (not link_info.empty())
            {
                auto sphere = min_sphere_of_spheres(link_info);

                SphereInfo info;
                info.geom_index = bs;
                info.radius = sphere[3];
                info.parent_joint = link_info[0].parent_joint;
                info.relative = SE3::Identity();
                info.relative.translation()[0] = sphere[0];
                info.relative.translation()[1] = sphere[1];
                info.relative.translation()[2] = sphere[2];

                bounding_spheres.emplace(i, info);
                bounding_sphere_index.emplace_back(bs);
                links_with_geometry.emplace_back(i);
                bs++;
            }
            else
            {
                bounding_sphere_index.emplace_back(0);
            }
        }
    }

    auto collision_pair_to_frame_pair(const CollisionPair &cp) -> std::pair<std::size_t, std::size_t>
    {
        const auto &geom1 = collision_model.geometryObjects[cp.first];
        const auto &geom2 = collision_model.geometryObjects[cp.second];

        std::size_t link1_idx = geom1.parentFrame;
        std::size_t link2_idx = geom2.parentFrame;

        return std::make_pair(std::min(link1_idx, link2_idx), std::max(link1_idx, link2_idx));
    }

    auto extract_collision_data() -> void
    {
        for (const auto &cp : collision_model.collisionPairs)
        {
            allowed_link_pairs.insert(collision_pair_to_frame_pair(cp));
        }
    }

    auto get_adjacent_frames() -> std::set<std::pair<std::size_t, std::size_t>>
    {
        const auto nf = model.frames.size();
        const auto nj = model.joints.size();

        std::set<std::pair<std::size_t, std::size_t>> adjacents;

        for (auto i = 0U; i < nf; ++i)
        {
            for (auto j = i + 1; j < nf; ++j)
            {
                const auto &frame_i = model.frames[i];
                const auto &frame_j = model.frames[j];

                if (frame_i.parentJoint < nj and frame_j.parentJoint < nj)
                {
                    const auto &joint_i = model.joints[frame_i.parentJoint];
                    const auto &joint_j = model.joints[frame_j.parentJoint];

                    // Check if joints are parent-child related
                    if (model.parents[frame_i.parentJoint] == frame_j.parentJoint or
                        model.parents[frame_j.parentJoint] == frame_i.parentJoint)
                    {
                        adjacents.insert({i, j});
                    }
                }
            }
        }

        return adjacents;
    }

    auto guess_self_collisions(std::size_t n = 1000000U) -> void
    {
        collision_model.addAllCollisionPairs();

        Data data(model);
        GeometryData collision_data(collision_model);

        std::set<std::pair<std::size_t, std::size_t>> always_pairs;

        for (auto j = 0U; j < collision_model.collisionPairs.size(); ++j)
        {
            always_pairs.emplace(collision_pair_to_frame_pair(collision_model.collisionPairs[j]));
        }

        allowed_link_pairs.clear();

        for (auto i = 0U; i < n; ++i)
        {
            auto q = randomConfiguration(model);
            computeCollisions(model, data, collision_model, collision_data, q);

            for (auto j = 0U; j < collision_model.collisionPairs.size(); ++j)
            {
                const auto &cr = collision_data.collisionResults[j];
                auto pair = collision_pair_to_frame_pair(collision_model.collisionPairs[j]);

                if (cr.isCollision())
                {
                    allowed_link_pairs.insert(pair);
                }
                else
                {
                    auto it = always_pairs.find(pair);
                    if (it != always_pairs.end())
                    {
                        always_pairs.erase(it);
                    }
                }
            }
        }

        // Remove all adjacent frames
        auto adjacents = get_adjacent_frames();
        for (const auto &pair : adjacents)
        {
            allowed_link_pairs.erase(pair);
        }

        // Remove all pairs that never collided
        for (const auto &pair : always_pairs)
        {
            allowed_link_pairs.erase(pair);
        }

        // Add remaining potential collisions
        collision_model.removeAllCollisionPairs();
        for (const auto &pair : allowed_link_pairs)
        {
            collision_model.addCollisionPair(CollisionPair(pair.first, pair.second));
        }
    }

    Model model;
    GeometryModel collision_model;
    std::string end_effector_name;
    std::size_t end_effector_index;

    float min_radius{std::numeric_limits<float>::max()};
    float max_radius{std::numeric_limits<float>::min()};
    std::vector<SphereInfo> spheres;
    std::map<std::size_t, SphereInfo> bounding_spheres;
    std::vector<std::size_t> links_with_geometry;
    std::vector<std::vector<std::size_t>> per_link_spheres;
    std::set<std::pair<std::size_t, std::size_t>> allowed_link_pairs;
    std::vector<std::size_t> bounding_sphere_index;
};

auto trace_sphere(const SphereInfo &sphere, const ADData &ad_data, ADVectorXs &data, std::size_t index)
{
    const auto &joint_placement = ad_data.oMi[sphere.parent_joint];

    Eigen::Matrix<ADCG, 3, 1> local_translation;
    local_translation[0] = sphere.relative.translation()[0];
    local_translation[1] = sphere.relative.translation()[1];
    local_translation[2] = sphere.relative.translation()[2];

    Eigen::Matrix<ADCG, 3, 1> world_position =
        joint_placement.rotation() * local_translation + joint_placement.translation();

    data[index + 0] = world_position[0];
    data[index + 1] = world_position[1];
    data[index + 2] = world_position[2];
    data[index + 3] = ADCG(sphere.radius);
}

auto trace_frame(std::size_t ee_index, const ADData &ad_data, ADVectorXs &data, std::size_t index)
{
    const auto &oMf = ad_data.oMf[ee_index];

    data[index + 0] = oMf.translation()[0];
    data[index + 1] = oMf.translation()[1];
    data[index + 2] = oMf.translation()[2];

    const auto &R = oMf.rotation();

    // Eigen stores as column major
    data[index + 3] = R(0, 0);
    data[index + 4] = R(1, 0);
    data[index + 5] = R(2, 0);
    data[index + 6] = R(0, 1);
    data[index + 7] = R(1, 1);
    data[index + 8] = R(2, 1);
    data[index + 9] = R(0, 2);
    data[index + 10] = R(1, 2);
    data[index + 11] = R(2, 2);
}

struct Traced
{
    std::string code;
    std::size_t temp_variables;
    std::size_t outputs;
};

auto trace_sphere_cc_fk(
    const RobotInfo &info,
    bool spheres = true,
    bool bounding_spheres = true,
    bool fk = true) -> Traced
{
    auto nq = info.model.nq;
    ADModel ad_model = info.model.cast<ADCG>();
    ADData ad_data(ad_model);

    ADVectorXs ad_q(nq);
    for (auto i = 0U; i < nq; ++i)
    {
        ad_q[i] = ADCG(0.0);
    }

    Independent(ad_q);

    forwardKinematics(ad_model, ad_data, ad_q);
    updateFramePlacements(ad_model, ad_data);

    std::size_t n_spheres_data = (spheres) ? info.spheres.size() * 4 : 0;
    std::size_t n_bounding_spheres_data = (bounding_spheres) ? info.bounding_spheres.size() * 4 : 0;
    std::size_t n_fk_data = (fk) ? 12 : 0;
    std::size_t n_out = n_spheres_data + n_bounding_spheres_data + n_fk_data;

    ADVectorXs data(n_out);

    if (spheres)
    {
        for (auto i = 0U; i < info.spheres.size(); ++i)
        {
            const auto &sphere = info.spheres[i];
            trace_sphere(sphere, ad_data, data, sphere.geom_index * 4);
        }
    }

    if (bounding_spheres)
    {
        for (auto i = 0U; i < info.model.frames.size(); ++i)
        {
            auto sphere_it = info.bounding_spheres.find(i);
            if (sphere_it != info.bounding_spheres.end())
            {
                const auto &sphere = sphere_it->second;
                trace_sphere(sphere, ad_data, data, sphere.geom_index * 4 + n_spheres_data);
            }
        }
    }

    if (fk)
    {
        trace_frame(info.end_effector_index, ad_data, data, n_spheres_data + n_bounding_spheres_data);
    }

    // Create the AD function
    ADFun<CGD> collision_sphere_func(ad_q, data);

    CodeHandler<double> handler;
    CppAD::vector<CGD> ind_vars(nq);
    handler.makeVariables(ind_vars);

    CppAD::vector<CGD> result = collision_sphere_func.Forward(0, ind_vars);

    LanguageCCustom<double> langC("double");
    LangCDefaultVariableNameGenerator<double> nameGen;

    std::ostringstream function_code;
    handler.generateCode(function_code, langC, result, nameGen);

    return Traced{function_code.str(), handler.getTemporaryVariableCount(), n_out};
}


auto trace_tsr_error_function(
    const RobotInfo &info
    ) -> Traced
{

    const double DT = 0.1;
    const double damp = 1e-6;
    auto nq = info.model.nq;
    auto nv = info.model.nv;
    const size_t nt = 6; // task space is se3
    // const size_t ntnt = 16; // for 4x4 matrix

    ADModel ad_model = info.model.cast<ADCG>();
    ADData ad_data(ad_model);

    // Total inputs is:
    // 2 * 4x4 matrices for constraint space
    // 2 * 6 bounds for constraint space
    // nq  for configuration space 
    const size_t num_inp = 7 * 2 + nt * 2 + nq;


    ADVectorXs ad_inp(num_inp); // 3 4x4 matrices
    for (auto i = 0U; i < num_inp; ++i)
        ad_inp[i] = ADCG(0.0);

    Independent(ad_inp);

    Eigen::Vector3<ADCG> rTep;
    Eigen::Vector3<ADCG> wTrp;

    ADVectorXs lb(nt);
    ADVectorXs ub(nt);
    ADVectorXs ad_q(nq);


    // Copying inputs from ad_inp into individual matrices
    for (auto i = 0U; i < nq; i++)
        ad_q[i] = ad_inp[i]; // This is the first 7 vars for nq


    Eigen::Quaternion<ADCG> rTeq(ad_inp[nq + 0 + 0], ad_inp[nq + 0 + 1], ad_inp[nq + 0 + 2], ad_inp[nq + 0 + 3]); // Next 7 for rTe
    Eigen::Quaternion<ADCG> wTrq(ad_inp[nq + 7 + 0], ad_inp[nq + 7 + 1], ad_inp[nq + 7 + 2], ad_inp[nq + 7 + 3]); // 7 after that for wTr


    for (auto i=0U; i < 3; i++)
    {
        rTep[i] = ad_inp[nq + 4 + i];
        wTrp[i] = ad_inp[nq + 7 + 4 + i];
    }


    for (auto i = 0U; i < nt; i++)
    {
        lb[i] = ad_inp[nq +  2 * 7 + i];
        ub[i] = ad_inp[nq + 2 * 7 + nt + i];
    }

    SE3Tpl<ADCG, 0> rTe (rTeq, rTep); // it is assumed that this err is expressed in the eef joint frame 
    SE3Tpl<ADCG, 0> wTr (wTrq, wTrp);  
    // input setup done


    forwardKinematics(ad_model, ad_data, ad_q);
    updateFramePlacements(ad_model, ad_data);


    // compute error term
    const auto wTobj = ad_data.oMf[info.end_effector_index] * rTe.inverse();
    const auto rTobj = wTr.inverse() * wTobj;


    ADVectorXs displacement(nt);
    displacement.setZero();
    Eigen::AngleAxis<ADCG> aa_error; //(rTobj.rotation_impl());
    displacement << rTobj.translation_impl(), aa_error.axis() * aa_error.angle();


    std::size_t n_out = nt;
    ADVectorXs data(n_out);
    for (auto i = 0U; i < nt; i++)
        // data[i] = min(displacement[i] - lb[i], displacement[i] * 0.) + max(displacement[i] - ub[i], displacement[i] * 0.);
        data[i] = displacement[i];



    // Create the AD function
    ADFun<CGD> jacobian_error_func(ad_inp, data);
    CodeHandler<double> handler;
    CppAD::vector<CGD> ind_vars(num_inp);
    handler.makeVariables(ind_vars);

    CppAD::vector<CGD> result = jacobian_error_func.Forward(0, ind_vars);

    CppAD::vector<CGD> jacobian_ind_vars(num_inp);

    for (auto i=0U; i < num_inp; i++)
    {
        if (i < nq)
            jacobian_ind_vars[i] = ind_vars[i];
        else
            jacobian_ind_vars[i] = ind_vars[i] * 0.0;
    }
    
    std::cout << "Computing jacobian " << std::endl;
    CppAD::vector<CGD> jac = jacobian_error_func.Jacobian(jacobian_ind_vars);
    std::cout << "Computed jacobian " << std::endl;

    CppAD::vector<CGD> jac_e_q(n_out * nq);

    for(auto i=0U; i < n_out; i++)
    {
        for(auto j=0U; j < nq; j++)
        {
            jac_e_q[i * nq + j] = jac[i * num_inp + j];
        }
    }


    std::cout << "Copied jacobian " << std::endl;
    // std::move(jac_e_q.begin(), jac_e_q.end(), std::back_inserter(result));
    std::move(result.begin(), result.end(), std::back_inserter(jac_e_q));


    LanguageCCustom<double> langC("double");
    LangCDefaultVariableNameGenerator<double> nameGen;

    std::ostringstream function_code;
    handler.generateCode(function_code, langC, jac_e_q, nameGen);

    return Traced{function_code.str(), handler.getTemporaryVariableCount(), jac_e_q.size()};
}


auto trace_solve_tsr_function(
    const RobotInfo &info
    ) -> Traced
{

    const double DT = 0.1;
    const double damp = 1e-6;
    auto nq = info.model.nq;
    auto nv = info.model.nv;
    const size_t nt = 6; // task space is se3


    const size_t num_inp = nt + nt * nq;
    ADVectorXs ad_inp(num_inp); // 3 4x4 matrices
    for (auto i = 0U; i < num_inp; ++i)
        ad_inp[i] = ADCG(0.0);

    Independent(ad_inp);

    ADVectorXs ad_e(nt);
    ADMatrixXs ad_J(nt, nq);

    // Copying inputs from ad_inp into individual matrices
    for (auto i = 0U; i < nt; i++)
        ad_e[i] = ad_inp[i]; // This is the first 7 vars for nq

    // Copying inputs from ad_inp into individual matrices
    ad_J = Eigen::Map<ADMatrixXs>(&ad_inp[nt], nt, nq);

    ADVectorXs grad(nq);
    // ADMatrixXs decomposed(nt, nt);
    auto decomposed = (ad_J * ad_J.transpose()).ldlt().matrixLDLT();
    std::size_t n_out = nt * nt;
    ADVectorXs data(n_out);


    // grad = ad_J.transpose() * (ad_J * ad_J.transpose()).llt().solve(ad_e);

    // std::size_t n_out = nt * nt;
    // ADVectorXs data(n_out);

    for (auto i = 0U; i < nt; i++)
        for (auto j = 0U; j < nt; j++)
            data[i * nt + j] = decomposed(i, j);

    // Create the AD function
    ADFun<CGD> solve_func(ad_inp, data);
    CodeHandler<double> handler;
    CppAD::vector<CGD> ind_vars(num_inp);
    handler.makeVariables(ind_vars);

    CppAD::vector<CGD> result = solve_func.Forward(0, ind_vars);


    LanguageCCustom<double> langC("double");
    LangCDefaultVariableNameGenerator<double> nameGen;

    std::ostringstream function_code;
    handler.generateCode(function_code, langC, result, nameGen);

    return Traced{function_code.str(), handler.getTemporaryVariableCount(), result.size()};
}


// if\s*\(\s*(v\[\d+\])\s*<\s*([0-9]*\.?[0-9]*)\s*\)\n\s*\{\s*\1\s*=\s*\2;\n\s*\}\n\s*else\s*\n\s*\{\s*\1\s*=\s*\1;\n\s*\} --> $1 = max($1, $2);
// if\s*\(\s*([0-9]*\.?[0-9]*)\s*<\s*(v\[\d+\])\s*\)\n\s*\{\s*\2\s*=\s*\1;\n\s*\}\n\s*else\s*\n\s*\{\s*\2\s*=\s*\2;\n\s*\} --> $2 = min($2, $1);

// for minmax 
// if\s*\(\s*(v\[\d+\])\s*<\s*([0-9]*\.?[0-9]*)\s*\)\n\s*\{\s*\1\s*=\s*\2;\n\s*\}\n\s*else\s*\n\s*\{\s*\1\s*=\s*([0-9]*\.?[0-9]*);\n\s*\} --> $1 = max($2, min($1, $3));
// if\s*\(\s*([0-9]*\.?[0-9]*)\s*<\s*(v\[\d+\])\s*\)\n\s*\{\s*\2\s*=\s*\1;\n\s*\}\n\s*else\s*\n\s*\{\s*\2\s*=\s*([0-9]*\.?[0-9]*);\n\s*\} --> $2 = min(max($2, $3), $1);


// for generic
// if\s*\((.*?)\)\n\s*\{\s*(\w+(?:\[\d+\])?)\s*=\s*(.*?);\s*\}\n\s*else\s*\n\s*\{\s*\2\s*=\s*(.*?);\s*\} --> $2 = ($1) ? $3 : $4;
// if\s*\(\s*(.*?)\s*<\s*(.*?)\s*\)\s*\{\s*(\w+(?:\[\d+\])?)\s*=\s*(.*?);\s*\}\n\s*else\s*\n\s*\{\s*\2\s*=\s*(.*?);\s*\} --> $3 = ($1 < $2) ? $4 : $5;

// $

int main(int argc, char **argv)
{
    cxxopts::Options options(argv[0], "Tracing compiler for forward kinematics and collision checking");

    options.positional_help("[JSON configuration filename]").show_positional_help();

    options.add_options()                                                                       //
        ("f,configuration_file", "JSON configuration filename", cxxopts::value<std::string>())  //
        ("o,output_filename", "Output JSON filename", cxxopts::value<std::string>())            //
        ("t,output_template",
         "Output template filename (override configuration file)",
         cxxopts::value<std::string>())  //
        ("h,help", "Print usage")        //
        ;

    options.parse_positional({"configuration_file"});

    auto result = options.parse(argc, argv);

    if (result.count("help"))
    {
        std::cout << options.help() << std::endl;
        exit(0);
    }

    if (not result.count("configuration_file"))
    {
        throw std::runtime_error(fmt::format("Must provide configuration file!"));
    }

    std::filesystem::path json_path(result["configuration_file"].as<std::string>());
    auto parent_path = json_path.parent_path();

    if (not std::filesystem::exists(json_path))
    {
        throw std::runtime_error(fmt::format("JSON file {} does not exist!", json_path.string()));
    }

    if (not std::filesystem::exists(json_path))
    {
    }

    std::ifstream json_file(json_path);
    nlohmann::json data;

    try
    {
        data = nlohmann::json::parse(json_file);
    }
    catch (std::exception &e)
    {
        throw std::runtime_error(fmt::format("Failed to parse JSON file! Error: \n{}", e.what()));
    }

    std::optional<std::filesystem::path> srdf_path = {};
    if (data.contains("srdf"))
    {
        srdf_path = parent_path / data["srdf"];
    }

    std::optional<std::string> end_effector_name = {};
    if (data.contains("end_effector"))
    {
        end_effector_name = data["end_effector"];
    }

    RobotInfo robot(parent_path / data["urdf"], srdf_path, end_effector_name);

    data.update(robot.json());

    auto traced_eefk_code = trace_sphere_cc_fk(robot, false, false, true);
    data["eefk_code"] = traced_eefk_code.code;
    data["eefk_code_vars"] = traced_eefk_code.temp_variables;
    data["eefk_code_output"] = traced_eefk_code.outputs;

    auto traced_spherefk_code = trace_sphere_cc_fk(robot, true, false, false);
    data["spherefk_code"] = traced_spherefk_code.code;
    data["spherefk_code_vars"] = traced_spherefk_code.temp_variables;
    data["spherefk_code_output"] = traced_spherefk_code.outputs;

    auto traced_ccfk_code = trace_sphere_cc_fk(robot, true, true, false);
    data["ccfk_code"] = traced_ccfk_code.code;
    data["ccfk_code_vars"] = traced_ccfk_code.temp_variables;
    data["ccfk_code_output"] = traced_ccfk_code.outputs;

    auto traced_ccfkee_code = trace_sphere_cc_fk(robot, true, true, true);
    data["ccfkee_code"] = traced_ccfkee_code.code;
    data["ccfkee_code_vars"] = traced_ccfkee_code.temp_variables;
    data["ccfkee_code_output"] = traced_ccfkee_code.outputs;

    auto traced_tsr_error_function_code  = trace_tsr_error_function(robot);
    data["tsr_error_function_code"] = traced_tsr_error_function_code.code;
    data["tsr_error_function_code_vars"] = traced_tsr_error_function_code.temp_variables;
    data["tsr_error_function_code_output"] = traced_tsr_error_function_code.outputs;

    auto traced_solve_tsr_function_code  = trace_solve_tsr_function(robot);
    data["solve_tsr_function_code"] = traced_solve_tsr_function_code.code;
    data["solve_tsr_function_code_vars"] = traced_solve_tsr_function_code.temp_variables;
    data["solve_tsr_function_code_output"] = traced_solve_tsr_function_code.outputs;


    inja::Environment env;

    for (const auto &subt : data["subtemplates"])
    {
        inja::Template temp = env.parse_template(parent_path / subt["template"]);
        env.include_template(subt["name"], temp);
    }

    std::string output_template;
    if (result.count("output_template"))
    {
        output_template = result["output_template"].as<std::string>();
    }
    else
    {
        output_template = data["output"];
    }

    inja::Template temp = env.parse_template(parent_path / data["template"]);
    env.write(temp, data, output_template);

    std::string output_filename;
    if (result.count("output_filename"))
    {
        output_filename = result["output_filename"].as<std::string>();
    }
    else
    {
        output_filename = "output.json";
    }

    std::ofstream output_file(output_filename);
    output_file << data.dump();
    output_file.close();

    return 0;
}
