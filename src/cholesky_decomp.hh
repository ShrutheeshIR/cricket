#include <stdlib.h>
#include <math.h>

template <typename Matrix, typename scalar_type>
Matrix cholesky_factor(const Matrix& input)
{
    size_t n = input.rows();
    Matrix result(n, n);

    for (size_t i = 0; i < n; ++i)
    {
        for (size_t k = 0; k < i; ++k) 
        {
            scalar_type value = input(i, k);
            for (size_t j = 0; j < k; ++j)
                value -= result(i, j) * result(k, j);

            result(i, k) = value / result(k, k);
        }
        scalar_type value = input(i, i);
        for (size_t j = 0; j < i; ++j)
            value -= result(i, j) * result(i, j);
        result(i, i) = sqrt(value);
    }
    return result;
}

template <typename Matrix, typename Vector, typename scalar_type>
Vector cholesky_solve(const Matrix& A, const Vector& B)
{
    size_t n = B.rows();
    Vector result(n);

    for (size_t i = 0; i < n; ++i)
    {
        auto value = B(i);
        for (size_t j = 0; j  < n; ++j)
            value -= A(i, j) * B(j);
        result(i) = value / A(i, i);
    }
    return result;

}
