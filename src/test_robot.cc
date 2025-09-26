#include "cholesky_decomp.hh"
#include "pinocchio/math/fwd.hpp"

#include <cmath>
#include <boost/mpl/int.hpp>
// #include <cppad/cg/support/cppadcg_eigen.hpp>

#include "pinocchio/autodiff/cppad.hpp"


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
// #include <inja/inja.hpp>
#include <cxxopts.hpp>

#include <filesystem>
#include <stdexcept>
#include <vector>
#include <optional>
// #include "lang_gen.hh"

using namespace pinocchio;
using namespace CppAD;
// using namespace CppAD::cg;

// Typedef for AD types
using CGD = double;
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


auto trace_full_tsr_project(
    const RobotInfo &info,
    ADVectorXs ad_inp
    ) -> CppAD::vector<CGD>
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

    Independent(ad_inp);

    Eigen::Vector3<ADCG> rTep;
    Eigen::Vector3<ADCG> wTrp;

    ADVectorXs lb(nt);
    ADVectorXs ub(nt);
    ADVectorXs ad_q(nq);

    // Copying inputs from ad_inp into individual matrices
    for (auto i = 0U; i < nq; i++)
        ad_q[i] = ad_inp[i]; // This is the first 7 vars for nq

    // print the q
    // for (auto i = 0U; i < nq; i++)
    //     std::cout << ad_q[i] << " ";
    // std::cout << std::endl;

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

    // std::cout << "rteq" << rTeq.coeffs().transpose() << " " << rTep.transpose() << std::endl;
    // std::cout << "wtrq" << wTrq.coeffs().transpose() << " " << wTrp.transpose() << std::endl;


    forwardKinematics(ad_model, ad_data, ad_q);
    updateFramePlacements(ad_model, ad_data);

    // std::cout << "rteq = \n" << rTe.toHomogeneousMatrix_impl() << std::endl;
    // std::cout << "wtrq = \n" << wTr.toHomogeneousMatrix_impl() << std::endl;
    // std::cout << "eef = \n" << ad_data.oMf[info.end_effector_index].toHomogeneousMatrix_impl() << std::endl;

    // compute error term
    const auto wTobj = ad_data.oMf[info.end_effector_index] * rTe.inverse();

    // const auto rTw = wTr.inverse(); //.matrix();
    // auto rTobj = rTw * wTobj.matrix();

    const auto rTobj = wTr.inverse() * wTobj;

    // std::cout << "rTobj = \n" << rTobj << std::endl;

    ADVectorXs displacement(nt);
    displacement.setZero();
    displacement << rTobj.translation_impl() * 50.0, log3(rTobj.rotation_impl()) * 20.0;
    // ADCG zero(0.0);

    // const auto iMd = ad_data.oMf[info.end_effector_index].actInv(wTobj);
    // std::cout << "oMe = \n" << ad_data.oMf[info.end_effector_index].toHomogeneousMatrix_impl() << std::endl;
    // std::cout << "iMd = \n" << iMd.toHomogeneousMatrix_impl() << std::endl;

    // auto displacement = log6(iMd); // in joint frame
    // std::cout << "displacement = " << displacement << std::endl;

    // ADVectorXs displacement(nt);
    // displacement.setZero();
    // displacement << iMd.translation_impl(), log3(iMd.rotation_impl());


    std::size_t n_out = nt;
    ADVectorXs data(n_out);
    for (auto i = 0U; i < nt; i++){
        // ;
        // data[i] = min(displacement[i] - lb[i], displacement[i] * 1e-6) + max(displacement[i] - ub[i], displacement[i] * 1e-6);
        data[i] = displacement[i];
        // std::cout << displacement[i] << " ";
    }
    // std::cout << std::endl;
    // for (auto i = 0U; i < n_out; i++)
    //     std::cout << "Error[" << i << "] = " << data[i] << std::endl;


    // Create the AD function
    ADFun<CGD> jacobian_error_func(ad_inp, data);
    CppAD::vector<CGD> ind_vars(num_inp);
    for (auto i=0U; i < num_inp; i++)
        ind_vars[i] = CppAD::Value(ad_inp[i]);
    CppAD::vector<CGD> result = jacobian_error_func.Forward(0, ind_vars);
    CppAD::vector<CGD> jac = jacobian_error_func.Jacobian(ind_vars);

    // print result
    std::cout << "Result: ";
    for (auto i=0U; i < result.size(); i++)
        std::cout << result[i] << " ";
    std::cout << std::endl;

    using CGDVectorXs = Eigen::Matrix<CGD, Eigen::Dynamic, 1>;
    using CGDMatrixXs = Eigen::Matrix<CGD, Eigen::Dynamic, Eigen::Dynamic>;
    CGDMatrixXs ad_J(nt, nq);

    for(auto i=0U; i < n_out; i++)
        for(auto j=0U; j < nq; j++)
            ad_J(i, j) = jac[i * num_inp + j];
    
    // std::cout < "Jacobian is " << ad_J << std::endl;
    // for (auto i=0U; i < nt; i++){
    //     for (auto j=0U; j < nq; j++)
    //         std::cout << ad_J(i, j) << ",";
    //     std::cout << std::endl;
    // }

    CGDMatrixXs identity(nt, nt);
    identity.setIdentity();
    CGDVectorXs ad_e(nt);

    // set up ad_e
    for (auto i=0U; i < nt; i++){
        if (i < 3)
            ad_e(i) = result[i]; // for position, we want to reduce the err in 0.1s
        else
            ad_e(i) = result[i]; // for rotation, we want to reduce the err in 1 step
    }

    // compute solution here directly. 
    auto decomposed = cholesky_factor<CGDMatrixXs, CGD>(ad_J * ad_J.transpose() + identity * 1e-4);
    CGDVectorXs grad = ad_J.transpose() * cholesky_solve<CGDMatrixXs, CGDVectorXs, CGD>(decomposed, ad_e);

    // CGDVectorXs grad = ad_J.transpose() * (ad_J * ad_J.transpose() + identity * 1e-4).llt().solve(ad_e);
    // CGDVectorXs grad = ad_J.transpose() * ad_e;

    CppAD::vector<CGD> grad_vec(nq);
    for (auto i=0U; i < nq; i++)
        grad_vec[i] = grad(i);

    std::move(result.begin(), result.end(), std::back_inserter(grad_vec));

    return grad_vec;


}

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

    // data.update(robot.json());
    // return 0;

    std::array<float, 6> lower_bound = {
        -0.01, -0.01, -0.03, -0.14, -0.14, -0.14
    };
    std::array<float, 6> upper_bound = {
        0.03, 0.01, 0.03, 0.14, 0.14, 0.14
    };

    // std::array<float, 7> target_pose = {
    //     0, 1, 0, 0, 0.543325, 0.570738, 0.121557
    // };
    // std::array<float, 7> in_hand_pose = {
    //     1, 0, 0, 0, 0, 0, 0
    // };
    std::array<float, 7> q_init = {
        0.56,1.20,0.0,0.0,0.0,1.57,1.64
    };


    Eigen::Matrix<float, 4, 4> T;
    // T <<   1,0,0, 0.543325, 0,-0.009, -0.999, 0.570738, 0, 0.999, -0.009, 0.121557, 0, 0, 0, 1;
    T <<   0.999,  0.   ,  0.037,  0.444,  0.037, -0.025, -0.999,  0.4, 0.001,  1.   , -0.025,  0.159,  0.   ,  0.   ,  0.   ,  1.  ;

    const Eigen::Transform<float, 3, Eigen::Isometry> target_pose(T);
    std::cout << "Target pose is : " << target_pose.translation().transpose() << std::endl;
    const auto in_hand_pose = Eigen::Transform<float, 3, Eigen::Isometry>::Identity();

    Eigen::Quaternion<float> q1(in_hand_pose.linear());
    std::array<float, 7> in_hand_pose_7 = {q1.w(), q1.x(), q1.y(), q1.z(), in_hand_pose.translation().x(), in_hand_pose.translation().y(), in_hand_pose.translation().z()};

    Eigen::Quaternion<float> q2(target_pose.linear());
    std::array<float, 7> target_pose_7 = {q2.w(), q2.x(), q2.y(), q2.z(), target_pose.translation().x(), target_pose.translation().y(), target_pose.translation().z()};



    // compose a new input of ADVectorXs ad_inp of q_init, target_pose, in_hand_pose, lower_bound, upper_bound
    ADVectorXs ad_inp(7 * 2 + 6 * 2 + robot.model.nq); // 3 4x4 matrices + 3 6D vectors + nq
    for (auto i = 0U; i < 7; ++i)
        ad_inp[i] = ADCG(q_init[i]);
    for (auto i = 0U; i < 7; ++i)
        ad_inp[7 + i] = ADCG(in_hand_pose_7[i]);
    for (auto i = 0U; i < 7; ++i)
        ad_inp[14 + i] = ADCG(target_pose_7[i]);
    for (auto i = 0U; i < 6; ++i)
        ad_inp[21 + i] = ADCG(lower_bound[i]);
    for (auto i = 0U; i < 6; ++i)
        ad_inp[27 + i] = ADCG(upper_bound[i]);


    for (auto i=0U; i < 10; i++){
        std::cout << "Iteration " << i << " : ";
        auto val = trace_full_tsr_project(robot, ad_inp);
        for (auto j=0U; j < robot.model.nq; j++){
            ad_inp[j] = ad_inp[j] - ADCG(1.0) * ADCG(val[j]);
            // clip by bounds
            if (CppAD::Value(ad_inp[j]) < robot.model.lowerPositionLimit[j])
                ad_inp[j] = ADCG(robot.model.lowerPositionLimit[j]);
            if (CppAD::Value(ad_inp[j]) > robot.model.upperPositionLimit[j])
                ad_inp[j] = ADCG(robot.model.upperPositionLimit[j]);
        }
    }
    std::cout << "Final q is : ";
    for (auto j=0U; j < robot.model.nq; j++)
        std::cout << CppAD::Value(ad_inp[j]) << " ";
    std::cout << std::endl;


    // for (auto i=0U; i < val.size(); i++)
    //     std::cout << val[i] << " ";
    // std::cout << std::endl;

    // int num_inp = 7 * 2 + 6 * 2 + robot.model.nq; // 3 4x4 matrices + 3 6D vectors + nq
    // ADVectorXs ad_inp(num_inp); // 3 4x4 matrices
    // for (auto i = 0U; i < num_inp; ++i)
    //     ad_inp[i] = ADCG(0.0);

    // auto new_q = ad_inp - val;

    // for (auto i=0U; i < robot.model.nq; i++) {
    //     ad_inp[i] = ad_inp[i] + val[i];
    //     std::cout << ad_inp[i] << " ";
    // }
    // std::cout << std::endl;

    
    // std::cout << "New q is : ";
    // for (auto i=0U; i < robot.model.nq; i++)
    //     std::cout << CppAD::Value(new_q[i]) << " ";
    // std::cout << std::endl;

    // val = trace_full_tsr_project(robot, ad_inp2);
    // for (auto i=0U; i < val.size(); i++)
    //     std::cout << val[i] << " ";
    // std::cout << std::endl;

    return 0;

}