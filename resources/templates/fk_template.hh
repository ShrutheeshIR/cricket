#pragma once

#include <vamp/vector.hh>
#include <vamp/vector/math.hh>
#include <vamp/collision/environment.hh>
#include <vamp/collision/validity.hh>

// NOLINTBEGIN(*-magic-numbers)
namespace vamp::robots
{
struct {{name}}
{
    static constexpr char* name = "{{lower(name)}}";
    static constexpr std::size_t dimension = {{n_q}};
    static constexpr std::size_t n_spheres = {{n_spheres}};
    static constexpr float min_radius = {{min_radius}};
    static constexpr float max_radius = {{max_radius}};
    static constexpr std::size_t resolution = {{resolution}};
    static constexpr std::size_t n_eef = {{num_end_effectors}};
    static constexpr std::size_t num_bounding_spheres = {{num_bounding_spheres}};

    static constexpr std::array<std::string_view, dimension> joint_names = {"{{join(joint_names, "\", \"")}}"};
    static constexpr std::array<std::string_view, {{num_end_effectors}}> end_effectors ={"{{join(end_effectors, "\", \"")}}"};

    using Configuration = FloatVector<dimension>;
    using ConfigurationArray = std::array<FloatT, dimension>;

    struct alignas(FloatVectorAlignment) ConfigurationBuffer
        : std::array<float, Configuration::num_scalars_rounded>
    {
    };

    template <std::size_t rake>
    using ConfigurationBlock = FloatVector<rake, dimension>;

    template <std::size_t rake>
    struct Spheres
    {
        FloatVector<rake, n_spheres> x;
        FloatVector<rake, n_spheres> y;
        FloatVector<rake, n_spheres> z;
        FloatVector<rake, n_spheres> r;
    };

    alignas(Configuration::S::Alignment) static constexpr std::array<float, dimension> s_m{
        {{join(bound_range, ", ")}}
    };

    alignas(Configuration::S::Alignment) static constexpr std::array<float, dimension> s_a{
        {{join(bound_lower, ", ")}}
    };

    alignas(Configuration::S::Alignment) static constexpr std::array<float, dimension> d_m{
        {{join(bound_descale, ", ")}}
    };

    static inline void scale_configuration(Configuration& q) noexcept
    {
        q = q * Configuration(s_m) + Configuration(s_a);
    }

    static inline void descale_configuration(Configuration& q) noexcept
    {
        q = (q - Configuration(s_a)) * Configuration(d_m);
    }

    template <std::size_t rake>
    static inline void scale_configuration_block(ConfigurationBlock<rake> &q) noexcept
    {
        {% for index in range(n_q) -%}
        q[{{index}}] = {{ at(bound_lower, index) }} + (q[{{index}}] * {{ at(bound_range, index) }});
        {%- endfor %}
    }

    template <std::size_t rake>
    static inline void descale_configuration_block(ConfigurationBlock<rake> & q) noexcept
    {
        {% for index in range(n_q) -%}
        q[{{index}}] = {{ at(bound_descale, index) }} * (q[{{index}}] - {{ at(bound_lower, index) }});
        {%- endfor %}
    }

    inline static auto space_measure() noexcept -> float
    {
        return {{measure}};
    }

    template <std::size_t rake>
    static inline void sphere_fk(const ConfigurationBlock<rake> &x, Spheres<rake> &out) noexcept
    {
        std::array<FloatVector<rake, 1>, {{spherefk_code_vars}}> v;
        std::array<FloatVector<rake, 1>, {{spherefk_code_output}}> y;

        {{spherefk_code}}

        for (auto i = 0U; i < {{n_spheres}}; ++i)
        {
            out.x[i] = y[i * 4 + 0];
            out.y[i] = y[i * 4 + 1];
            out.z[i] = y[i * 4 + 2];
            out.r[i] = y[i * 4 + 3];
        }
    }

    using Debug = std::pair<std::vector<std::vector<std::string>>, std::vector<std::pair<std::size_t, std::size_t>>>;

    template <std::size_t rake>
        static inline auto fkcc_debug(
            const vamp::collision::Environment<FloatVector<rake>> &environment,
            const ConfigurationBlock<rake> &x) noexcept -> Debug
    {
        std::array<FloatVector<rake, 1>, {{ccfk_code_vars}}> v;
        std::array<FloatVector<rake, 1>, {{ccfk_code_output}}> y;

        {{ccfk_code}}

        Debug output;

        {% for i in range(n_spheres) %}
        output.first.emplace_back(
            sphere_environment_get_collisions<decltype(x[0])>(
                environment,
                y[{{ i * 4 + 0 }}],
                y[{{ i * 4 + 1 }}],
                y[{{ i * 4 + 2 }}],
                y[{{ i * 4 + 3 }}]));
        {% endfor %}

        {% for i in range(length(allowed_link_pairs)) %}
        {% set pair = at(allowed_link_pairs, i) %}
        {% set link_1_index = at(pair, 0) %}
        {% set link_2_index = at(pair, 1) %}
        {% set link_1_spheres = at(per_link_spheres, link_1_index) %}
        {% set link_2_spheres = at(per_link_spheres, link_2_index) %}

        {% for j in range(length(link_1_spheres)) %}
        {% for k in range(length(link_2_spheres)) %}

        {% set sphere_1_loc = at(link_1_spheres, j) %}
        {% set sphere_2_loc = at(link_2_spheres, k) %}

        if (sphere_sphere_self_collision<decltype(x[0])>(y[{{ sphere_1_loc * 4 + 0}} ],
                                                         y[{{ sphere_1_loc * 4 + 1}} ],
                                                         y[{{ sphere_1_loc * 4 + 2}} ],
                                                         y[{{ sphere_1_loc * 4 + 3}} ],
                                                         y[{{ sphere_2_loc * 4 + 0}} ],
                                                         y[{{ sphere_2_loc * 4 + 1}} ],
                                                         y[{{ sphere_2_loc * 4 + 2}} ],
                                                         y[{{ sphere_2_loc * 4 + 3}} ]))
        {
            output.second.emplace_back({{ sphere_1_loc }}, {{ sphere_2_loc }});
        }

        {% endfor %}
        {% endfor %}
        {% endfor %}

        return output;
    }

    template <std::size_t rake>
        static inline bool fkcc(
            const vamp::collision::Environment<FloatVector<rake>> &environment,
            const ConfigurationBlock<rake> &x) noexcept
    {
        std::array<FloatVector<rake, 1>, {{ccfk_code_vars}}> v;
        std::array<FloatVector<rake, 1>, {{ccfk_code_output}}> y;

        {{ccfk_code}}
        {% include "ccfk" %}

        return true;
    }

    template <std::size_t rake>
    static inline bool fkcc_attach(
        const vamp::collision::Environment<FloatVector<rake>> &environment,
        const ConfigurationBlock<rake> &x) noexcept
    {
        std::array<FloatVector<rake, 1>, {{ccfkee_code_vars}}> v;
        std::array<FloatVector<rake, 1>, {{ccfkee_code_output}}> y;

        {{ccfkee_code}}
        {% include "ccfk" %}

        // attaching at {{ end_effectors }}
        // provide the eef pose of all end effectors.
        set_attachment_pose(environment, to_isometries<{{num_end_effectors}}>(&y[{{ccfkee_code_output - 12 * num_end_effectors}}]));

        //
        // attachment vs. environment collisions
        //
        if (attachment_environment_collision(environment))
        {
            return false;
        }

        //
        // attachment vs. robot collisions
        //

        {% for i in range(length(end_effector_collisions)) %}
        {% set link_index = at(end_effector_collisions, i) %}
        {% set link_bs = at(bounding_sphere_index, link_index) %}
        {% set link_spheres = at(per_link_spheres, link_index) %}

        // Attachment vs. {{ at(link_names, link_index )}}
        if (attachment_sphere_collision<decltype(x[0])>(environment,
                                                        y[{{(n_spheres + link_bs) * 4 + 0}}],
                                                        y[{{(n_spheres + link_bs) * 4 + 1}}],
                                                        y[{{(n_spheres + link_bs) * 4 + 2}}],
                                                        y[{{(n_spheres + link_bs) * 4 + 3}}]))
        {
            {% for j in range(length(link_spheres)) %}
            {% set sphere_index = at(link_spheres, j) %}
            if (attachment_sphere_collision<decltype(x[0])>(environment,
                                                            y[{{sphere_index * 4 + 0}}],
                                                            y[{{sphere_index * 4 + 1}}],
                                                            y[{{sphere_index * 4 + 2}}],
                                                            y[{{sphere_index * 4 + 3}}]))
            {
                return false;
            }
            {% endfor %}
        }
        {% endfor %}

        return true;
    }

    static inline auto eefk(const std::array<float, {{n_q}}> &x) noexcept -> std::array<Eigen::Isometry3f, {{num_end_effectors}}>
    {
        std::array<float, {{eefk_code_vars}}> v;
        std::array<float, {{eefk_code_output}}> y;

        {{eefk_code}}

        return to_isometries<{{num_end_effectors}}>(y.data());
    }

    template <std::size_t rake, typename InputVector, typename OutputVector>
    static inline auto tsr_error(const InputVector &x, OutputVector &out)
    {
        std::array<FloatVector<rake, 1>, {{tsr_error_code_vars}}> v;
        std::array<FloatVector<rake, 1>, {{tsr_error_code_output}}> y;

        {{tsr_error_code}}

        for(size_t i = 0; i < {{tsr_error_code_output}}; i++)
            out[i] = y[i];
    }

    template <std::size_t rake, typename InputVector, typename OutputVector>
    static inline auto solve_tsr_error_lm_inner(const InputVector &x, OutputVector &out)
    {
        std::array<FloatVector<rake, 1>, {{solve_tsr_error_lm_inner_code_vars}}> v;
        std::array<FloatVector<rake, 1>, {{solve_tsr_error_lm_inner_code_output}}> y;

        {{solve_tsr_error_lm_inner_code}}

        for(size_t i = 0; i < {{solve_tsr_error_lm_inner_code_output}}; i++)
            out[i] = y[i];
    }

    template <std::size_t rake, typename InputVector, typename OutputVector>
    static inline auto solve_tsr_error_lm_outer(const InputVector &x, OutputVector &out)
    {
        std::array<FloatVector<rake, 1>, {{solve_tsr_error_lm_outer_code_vars}}> v;
        std::array<FloatVector<rake, 1>, {{solve_tsr_error_lm_outer_code_output}}> y;

        {{solve_tsr_error_lm_outer_code}}

        for(size_t i = 0; i < {{solve_tsr_error_lm_outer_code_output}}; i++)
            out[i] = y[i];
    }

    template <std::size_t rake, typename InputVector, typename OutputVector>
    static inline auto solve_tsr_error_gradient_descent(const InputVector &x, OutputVector &out)
    {
        std::array<FloatVector<rake, 1>, {{solve_tsr_error_gradient_descent_code_vars}}> v;
        std::array<FloatVector<rake, 1>, {{solve_tsr_error_gradient_descent_code_output}}> y;

        {{solve_tsr_error_gradient_descent_code}}

        for(size_t i = 0; i < {{solve_tsr_error_gradient_descent_code_output}}; i++)
            out[i] = y[i];
    }


    template <std::size_t rake, typename InputVector, typename OutputVector>
    static inline auto compute_com(const InputVector &x, OutputVector &out)
    {
        std::array<FloatVector<rake, 1>, {{CoM_code_vars}}> v;
        std::array<FloatVector<rake, 1>, {{CoM_code_output}}> y;

        {{CoM_code}}

        for(size_t i = 0; i < {{CoM_code_output}}; i++)
            out[i] = y[i];
    }

    template <std::size_t rake, typename InputVector, typename OutputVector>
    static inline auto com_constraint_error(const InputVector &com_jac_polygons, size_t n, OutputVector &out)
    {
        // input com_jac_polygons is a 3 + (3 * robot.nq) + (n * 2) where n is the number of polygons

        const size_t polygon_offset = 3 + 3 * dimension;
        // initialize x, com_constraint_grad and out
        FloatVector<rake, 4 + 3> x;
        x[0] = com_jac_polygons[0];
        x[1] = com_jac_polygons[1];
        x[2] = com_jac_polygons[2];

        FloatVector<rake, 2 * 3> com_constraint_grad;
        com_constraint_grad[0] = 0.0;
        com_constraint_grad[1] = 0.0;
        com_constraint_grad[2] = 0.0;
        com_constraint_grad[3] = 0.0;
        com_constraint_grad[4] = 0.0;
        com_constraint_grad[5] = 0.0;


        // initialize the errors to be zero
        out[2 * dimension + 0] = 0.0;
        out[2 * dimension + 1] = 0.0;

        for(size_t polygon_idx = 0; polygon_idx < n; polygon_idx++){

            const auto polygon_next_idx = (polygon_idx + 1) % 4;
            x[3 + 0] = com_jac_polygons[polygon_offset + 2 * polygon_idx + 0];
            x[3 + 1] = com_jac_polygons[polygon_offset + 2 * polygon_idx + 1];
            x[3 + 2] = com_jac_polygons[polygon_offset + 2 * polygon_next_idx + 0];
            x[3 + 3] = com_jac_polygons[polygon_offset + 2 * polygon_next_idx + 1];


            std::array<FloatVector<rake, 1>, {{CoM_constraint_code_vars}}> v;
            std::array<FloatVector<rake, 1>, {{CoM_constraint_code_output}}> y;
            {{CoM_constraint_code}}

            // initialize_error
            out[2 * dimension + 0] = out[2 * dimension + 0] + y[0];
            out[2 * dimension + 1] = out[2 * dimension + 1] + y[1];

            for(size_t j = 0; j < 2 * 3; j++)
                com_constraint_grad[j] = com_constraint_grad[j] + y[2 + j];
        }

        // do the actual matrix multiplication
        for(size_t dim_idx = 0; dim_idx < dimension; dim_idx++){
            out[dim_idx] = com_constraint_grad[0] * com_jac_polygons[3 + dim_idx] + com_constraint_grad[1] * com_jac_polygons[3 + dimension + dim_idx] + com_constraint_grad[2] * com_jac_polygons[3 + 2 * dimension + dim_idx];
            out[dimension + dim_idx] = com_constraint_grad[3] * com_jac_polygons[3 + dim_idx] + com_constraint_grad[4] * com_jac_polygons[3 + dimension + dim_idx] + com_constraint_grad[5] * com_jac_polygons[3 + 2 * dimension + dim_idx];
        }
    }


    template <std::size_t rake, typename InputVector, typename OutputVector>
    static inline auto solve_com_function_lm_inner(const InputVector &x, OutputVector &out)
    {
        std::array<FloatVector<rake, 1>, {{solve_com_function_lm_inner_code_vars}}> v;
        std::array<FloatVector<rake, 1>, {{solve_com_function_lm_inner_code_output}}> y;

        {{solve_com_function_lm_inner_code}}

        for(size_t i = 0; i < {{solve_com_function_lm_inner_code_output}}; i++)
            out[i] = y[i];
    }

    template <std::size_t rake, typename InputVector, typename OutputVector>
    static inline auto solve_com_function_lm_outer(const InputVector &x, OutputVector &out)
    {
        std::array<FloatVector<rake, 1>, {{solve_com_function_lm_outer_code_vars}}> v;
        std::array<FloatVector<rake, 1>, {{solve_com_function_lm_outer_code_output}}> y;

        {{solve_com_function_lm_outer_code}}

        for(size_t i = 0; i < {{solve_com_function_lm_outer_code_output}}; i++)
            out[i] = y[i];
    }

    template <std::size_t rake, typename InputVector, typename OutputVector>
    static inline auto solve_com_function_gradient_descent(const InputVector &x, OutputVector &out)
    {
        std::array<FloatVector<rake, 1>, {{solve_com_function_gradient_descent_code_vars}}> v;
        std::array<FloatVector<rake, 1>, {{solve_com_function_gradient_descent_code_output}}> y;

        {{solve_com_function_gradient_descent_code}}

        for(size_t i = 0; i < {{solve_com_function_gradient_descent_code_output}}; i++)
            out[i] = y[i];
    }

    template <std::size_t rake, typename InputVector, typename OutputVector>
    static inline auto sphere_sphere_collision_error(const InputVector &x, OutputVector &out)
    {
        std::array<FloatVector<rake, 1>, {{sphere_sphere_collision_code_vars}}> v;
        std::array<FloatVector<rake, 1>, {{sphere_sphere_collision_code_output}}> y;

        {{sphere_sphere_collision_code}}

        for(size_t i = 0; i < {{sphere_sphere_collision_code_output}}; i++)
            out[i] = y[i];
    }

    template <std::size_t rake, typename InputVector, typename OutputVector>
    static inline auto sphere_cuboid_collision_error(const InputVector &x, OutputVector &out)
    {
        std::array<FloatVector<rake, 1>, {{sphere_cuboid_collision_code_vars}}> v;
        std::array<FloatVector<rake, 1>, {{sphere_cuboid_collision_code_output}}> y;

        {{sphere_cuboid_collision_code}}

        for(size_t i = 0; i < {{sphere_cuboid_collision_code_output}}; i++)
            out[i] = y[i];
    }

    template <std::size_t rake, typename InputSphereVector, typename InputSphVector>
    inline static auto sphere_sphere_collision_error(const InputSphereVector &x, InputSphVector &sph, FloatVector<rake, 4> &y)
    {
        std::array<FloatVector<rake, 1>, 5> v;

        v[0] = sph[0] - x[0];
        v[1] = sph[1] - x[1];
        v[2] = sph[2] - x[2];
        v[2] = (x[3] + sph[3]) * (x[3] + sph[3]) - (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
        y[0] = 0.5 * (v[2] + sqrt(1e-06 + v[2] * v[2]));
        v[2] = sph[0] - x[0];
        v[1] = sph[1] - x[1];
        v[0] = sph[2] - x[2];
        v[3] = (x[3] + sph[3]) * (x[3] + sph[3]) - (v[2] * v[2] + v[1] * v[1] + v[0] * v[0]);
        v[4] = (0.5 * 1. / sqrt(1e-06 + v[3] * v[3])) / 2.;
        v[4] = 0. - (0.5 + v[4] * v[3] + v[4] * v[3]);
        y[1] = 0. - (v[4] * v[2] + v[4] * v[2]);
        y[2] = 0. - (v[4] * v[1] + v[4] * v[1]);
        y[3] = 0. - (v[4] * v[0] + v[4] * v[0]);

    }

    template <std::size_t rake, typename InputSphereVector, typename InputCubVector>
    inline static auto sphere_cuboid_collision_error(const InputSphereVector &x, const InputCubVector &cub, FloatVector<rake, 4> &y)
    {
        std::array<FloatVector<rake, 1>, 18> v;

        v[0] = x[0] - cub[0];
        v[1] = x[1] - cub[1];
        v[2] = x[2] - cub[2];
        v[3] = v[0] * cub[3] + v[1] * cub[4] + v[2] * cub[5];
        v[3] = sqrt(1e-08 + v[3] * v[3]) - cub[12];
        v[3] = 0.5 * (v[3] + sqrt(1e-08 + v[3] * v[3]));
        v[4] = v[0] * cub[6] + v[1] * cub[7] + v[2] * cub[8];
        v[4] = sqrt(1e-08 + v[4] * v[4]) - cub[13];
        v[4] = 0.5 * (v[4] + sqrt(1e-08 + v[4] * v[4]));
        v[2] = v[0] * cub[9] + v[1] * cub[10] + v[2] * cub[11];
        v[2] = sqrt(1e-08 + v[2] * v[2]) - cub[14];
        v[2] = 0.5 * (v[2] + sqrt(1e-08 + v[2] * v[2]));
        v[2] = x[3] - sqrt(1e-08 + v[3] * v[3] + v[4] * v[4] + v[2] * v[2]);
        y[0] = 0.5 * (v[2] + sqrt(1e-08 + v[2] * v[2]));
        v[2] = x[0] - cub[0];
        v[4] = x[1] - cub[1];
        v[3] = x[2] - cub[2];
        v[1] = v[2] * cub[3] + v[4] * cub[4] + v[3] * cub[5];
        v[0] = sqrt(1e-08 + v[1] * v[1]);
        v[5] = v[0] - cub[12];
        v[6] = sqrt(1e-08 + v[5] * v[5]);
        v[7] = 0.5 * (v[5] + v[6]);
        v[8] = v[2] * cub[6] + v[4] * cub[7] + v[3] * cub[8];
        v[9] = sqrt(1e-08 + v[8] * v[8]);
        v[10] = v[9] - cub[13];
        v[11] = sqrt(1e-08 + v[10] * v[10]);
        v[12] = 0.5 * (v[10] + v[11]);
        v[3] = v[2] * cub[9] + v[4] * cub[10] + v[3] * cub[11];
        v[4] = sqrt(1e-08 + v[3] * v[3]);
        v[2] = v[4] - cub[14];
        v[13] = sqrt(1e-08 + v[2] * v[2]);
        v[14] = 0.5 * (v[2] + v[13]);
        v[15] = sqrt(1e-08 + v[7] * v[7] + v[12] * v[12] + v[14] * v[14]);
        v[16] = x[3] - v[15];
        v[17] = (0.5 * 1. / sqrt(1e-08 + v[16] * v[16])) / 2.;
        v[17] = ((0. - (0.5 + v[17] * v[16] + v[17] * v[16])) * 1. / v[15]) / 2.;
        v[14] = (v[17] * v[14] + v[17] * v[14]) * 0.5;
        v[13] = (v[14] * 1. / v[13]) / 2.;
        v[13] = ((v[14] + v[13] * v[2] + v[13] * v[2]) * 1. / v[4]) / 2.;
        v[13] = v[13] * v[3] + v[13] * v[3];
        v[12] = (v[17] * v[12] + v[17] * v[12]) * 0.5;
        v[11] = (v[12] * 1. / v[11]) / 2.;
        v[11] = ((v[12] + v[11] * v[10] + v[11] * v[10]) * 1. / v[9]) / 2.;
        v[11] = v[11] * v[8] + v[11] * v[8];
        v[17] = (v[17] * v[7] + v[17] * v[7]) * 0.5;
        v[6] = (v[17] * 1. / v[6]) / 2.;
        v[6] = ((v[17] + v[6] * v[5] + v[6] * v[5]) * 1. / v[0]) / 2.;
        v[6] = v[6] * v[1] + v[6] * v[1];
        y[1] = v[13] * cub[9] + v[11] * cub[6] + v[6] * cub[3];
        y[2] = v[13] * cub[10] + v[11] * cub[7] + v[6] * cub[4];
        y[3] = v[13] * cub[11] + v[11] * cub[8] + v[6] * cub[5];
    }

    template <std::size_t rake, typename InputVector>
    static inline auto spheres_fk_jac(const InputVector &x, std::array<FloatVector<rake, 1>, {{sphere_fk_positions_jac_code_output}}> &y)
    {
        std::array<FloatVector<rake, 1>, {{sphere_fk_positions_jac_code_vars}}> v;

        {{sphere_fk_positions_jac_code}}

    }


    template <std::size_t rake, typename OutputVector>
    static inline auto robot_spheres_collision_fn(const vamp::collision::Environment<FloatVector<rake>> &environment, const ConfigurationBlock<rake> &x, OutputVector &out)
    {
        std::array<FloatVector<rake, 1>, {{sphere_fk_positions_jac_code_vars}}> v;
        std::array<FloatVector<rake, 1>, {{sphere_fk_positions_jac_code_output}}> y;

        {{sphere_fk_positions_jac_code}}

        // now write my for loop and compute

        for(size_t sphere_idx = 0U; sphere_idx < n_spheres; sphere_idx++)
        {
            auto collision_penetration_distance = FloatVector<rake>::fill(0.0f);
            auto collision_penetration_jac = FloatVector<rake, 3>::fill(0.0f);

            for (const auto &es : environment.spheres)
            {

                // compute collision penetration distance and jacobian

                auto local_out = FloatVector<rake, 4>::fill(0.0f);
                sphere_sphere_collision_error(y[sphere_idx*4], es, local_out);

                collision_penetration_distance = collision_penetration_distance + local_out[0];
                collision_penetration_jac[0] = collision_penetration_jac[0] + local_out[1];
                collision_penetration_jac[1] = collision_penetration_jac[1] + local_out[2];
                collision_penetration_jac[2] = collision_penetration_jac[2] + local_out[3];
            }
            // now do sphere_cuboid_collision_code
            for (const auto &ec : environment.cuboids)
            {

                // compute collision penetration distance and jacobian

                auto local_out = FloatVector<rake, 4>::fill(0.0f);
                sphere_cuboid_collision_error(y[sphere_idx*4], ec, local_out);

                collision_penetration_distance = collision_penetration_distance + local_out[0];
                collision_penetration_jac[0] = collision_penetration_jac[0] + local_out[1];
                collision_penetration_jac[1] = collision_penetration_jac[1] + local_out[2];
                collision_penetration_jac[2] = collision_penetration_jac[2] + local_out[3];
            }

            // now propagate the jacobian properly
            for(auto i=0U; i < dimension; i++)
            {
                out[sphere_idx * dimension + i] =
                    collision_penetration_jac[0] * y[n_spheres * 4 + (3 * sphere_idx + 0) * dimension + i] +
                    collision_penetration_jac[1] * y[n_spheres * 4 + (3 * sphere_idx + 1) * dimension + i] +
                    collision_penetration_jac[2] * y[n_spheres * 4 + (3 * sphere_idx + 2) * dimension + i];
            }
            out[n_spheres * dimension + sphere_idx] = collision_penetration_distance;

        }
    }

    template <std::size_t rake, typename InputVector, typename OutputVector>
    static inline auto solve_sph_env_error_lm_inner(const InputVector &x, OutputVector &out)
    {
        std::array<FloatVector<rake, 1>, {{solve_sphere_env_function_lm_outer_code_vars}}> v;
        std::array<FloatVector<rake, 1>, {{solve_sphere_env_function_lm_outer_code_output}}> y;

        {{solve_sphere_env_function_lm_outer_code}}

        for(size_t i = 0; i < {{solve_sphere_env_function_lm_outer_code_output}}; i++)
            out[i] = y[i];
    }

    template <std::size_t rake, typename InputVector, typename OutputVector>
    static inline auto solve_sph_env_error_lm_outer(const InputVector &x, OutputVector &out)
    {
        std::array<FloatVector<rake, 1>, {{solve_sphere_env_function_lm_outer_code_vars}}> v;
        std::array<FloatVector<rake, 1>, {{solve_sphere_env_function_lm_outer_code_output}}> y;

        {{solve_sphere_env_function_lm_outer_code}}

        for(size_t i = 0; i < {{solve_sphere_env_function_lm_outer_code_output}}; i++)
            out[i] = y[i];
    }

    template <std::size_t rake, typename InputVector, typename OutputVector>
    static inline auto solve_sph_env_error_gradient_descent(const InputVector &x, OutputVector &out)
    {
        std::array<FloatVector<rake, 1>, {{solve_sphere_env_function_gradient_descent_code_vars}}> v;
        std::array<FloatVector<rake, 1>, {{solve_sphere_env_function_gradient_descent_code_output}}> y;

        {{solve_sphere_env_function_gradient_descent_code}}

        for(size_t i = 0; i < {{solve_sphere_env_function_gradient_descent_code_output}}; i++)
            out[i] = y[i];
    }


    {% if num_end_effectors > 1 %}
    template <std::size_t rake, typename InputVector, typename OutputVector>
    static inline auto tsr_bimanual_error(const InputVector &x, OutputVector &out)
    {
        std::array<FloatVector<rake, 1>, {{tsr_bimanual_error_code_vars}}> v;
        std::array<FloatVector<rake, 1>, {{tsr_bimanual_error_code_output}}> y;

        {{tsr_bimanual_error_code}}

        for(size_t i = 0; i < {{tsr_bimanual_error_code_output}}; i++)
            out[i] = y[i];
    }

    template <std::size_t rake, typename InputVector, typename OutputVector>
    static inline auto solve_tsr_relative_error_lm_inner(const InputVector &x, OutputVector &out)
    {
        std::array<FloatVector<rake, 1>, {{solve_relative_tsr_error_lm_inner_code_vars}}> v;
        std::array<FloatVector<rake, 1>, {{solve_relative_tsr_error_lm_inner_code_output}}> y;

        {{solve_relative_tsr_error_lm_inner_code}}

        for(size_t i = 0; i < {{solve_relative_tsr_error_lm_inner_code_output}}; i++)
            out[i] = y[i];
    }

    template <std::size_t rake, typename InputVector, typename OutputVector>
    static inline auto solve_tsr_relative_error_lm_outer(const InputVector &x, OutputVector &out)
    {
        std::array<FloatVector<rake, 1>, {{solve_relative_tsr_error_lm_outer_code_vars}}> v;
        std::array<FloatVector<rake, 1>, {{solve_relative_tsr_error_lm_outer_code_output}}> y;

        {{solve_relative_tsr_error_lm_outer_code}}

        for(size_t i = 0; i < {{solve_relative_tsr_error_lm_outer_code_output}}; i++)
            out[i] = y[i];
    }

    template <std::size_t rake, typename InputVector, typename OutputVector>
    static inline auto solve_tsr_relative_error_gradient_descent(const InputVector &x, OutputVector &out)
    {
        std::array<FloatVector<rake, 1>, {{solve_relative_tsr_error_gradient_descent_code_vars}}> v;
        std::array<FloatVector<rake, 1>, {{solve_relative_tsr_error_gradient_descent_code_output}}> y;

        {{solve_relative_tsr_error_gradient_descent_code}}

        for(size_t i = 0; i < {{solve_relative_tsr_error_gradient_descent_code_output}}; i++)
            out[i] = y[i];
    }

    template <std::size_t rake, typename InputVector, typename OutputVector>
    static inline auto solve_2_eef_tsr_error_lm_inner(const InputVector &x, OutputVector &out)
    {
        std::array<FloatVector<rake, 1>, {{solve_2_eef_tsr_error_lm_inner_code_vars}}> v;
        std::array<FloatVector<rake, 1>, {{solve_2_eef_tsr_error_lm_inner_code_output}}> y;

        {{solve_2_eef_tsr_error_lm_inner_code}}

        for(size_t i = 0; i < {{solve_2_eef_tsr_error_lm_inner_code_output}}; i++)
            out[i] = y[i];
    }

    template <std::size_t rake, typename InputVector, typename OutputVector>
    static inline auto solve_2_eef_tsr_error_lm_outer(const InputVector &x, OutputVector &out)
    {
        std::array<FloatVector<rake, 1>, {{solve_2_eef_tsr_error_lm_outer_code_vars}}> v;
        std::array<FloatVector<rake, 1>, {{solve_2_eef_tsr_error_lm_outer_code_output}}> y;

        {{solve_2_eef_tsr_error_lm_outer_code}}

        for(size_t i = 0; i < {{solve_2_eef_tsr_error_lm_outer_code_output}}; i++)
            out[i] = y[i];
    }

    template <std::size_t rake, typename InputVector, typename OutputVector>
    static inline auto solve_2_eef_tsr_error_gradient_descent(const InputVector &x, OutputVector &out)
    {
        std::array<FloatVector<rake, 1>, {{solve_2_eef_tsr_error_gradient_descent_code_vars}}> v;
        std::array<FloatVector<rake, 1>, {{solve_2_eef_tsr_error_gradient_descent_code_output}}> y;

        {{solve_2_eef_tsr_error_gradient_descent_code}}

        for(size_t i = 0; i < {{solve_2_eef_tsr_error_gradient_descent_code_output}}; i++)
            out[i] = y[i];
    }


    {% endif %}


};
}

// NOLINTEND(*-magic-numbers)
