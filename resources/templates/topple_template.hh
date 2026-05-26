#pragma once

// NOLINTBEGIN(*-magic-numbers)
namespace vamp::robots
{
struct {{name}}
{

    template <typename InputVector, typename OutputVector>
    static inline auto topple_nn_forward(const InputVector &x, OutputVector &out) noexcept
    {
        FloatVector<8, {{forward_code_vars}}> v;
        FloatVector<8, {{forward_code_output}}> y;
        {{forward_code}}

        for(auto i=0U;i<{{forward_code_output}};i++)
            out[i] = y[i];

    }

    template <typename InputVector, typename OutputVector>
    static inline auto topple_nn_time_forward_backward(const InputVector &x, OutputVector &out) noexcept
    {
        FloatVector<8, {{forward_and_backward_code_vars}}> v;
        FloatVector<8, {{forward_and_backward_code_output}}> y;
        {{forward_and_backward_code}}

        for(auto i=0U;i<{{forward_and_backward_code_output}};i++)
            out[i] = y[i];

    }



};
}

// NOLINTEND(*-magic-numbers)
