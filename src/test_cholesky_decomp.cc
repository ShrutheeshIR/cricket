#include "cholesky_decomp.hh"
#include <Eigen/Dense>
#include <iostream>
#include <stdexcept>

int main() {
    Eigen::MatrixXd A(6, 7);

    Eigen::VectorXd b(6);
    b << 0.244468 ,0.260982, -0.0285135, 1.75411, -0.0275496, 0.495869;
    // A << 2, 1, 0,
    //      1, 2, 1,
    //      0, 1, 2; // A is a positive definite matrix
    // A << -0.405167 0.075288 -0.10593 -0.0430207 -0.10593 0.143816 0 -0.017503 -0.812198 -0.00457613 0.48732 -0.00457613 0.156431 0 -0.702591 -0.0231952 -0.183691 0.0126756 -0.183691 -0.086861 0 0.242297 -0.572629 0.832758 0.572629 0.832758 0.572629 0.0631314 0.683943 -0.647626 -0.372322 0.647626 -0.372322 0.647626 -0.878269 -0.896306 -0.643406 -0.601699 0.643406 -0.601699 0.643406 0.728212; // A is a positive definite matrix
    A << -0.405167,0.075288,-0.10593,-0.0430207,-0.10593,0.143816,0,-0.017503,-0.812198,-0.00457613,0.48732,-0.00457613,0.156431,0,-0.702591,-0.0231952,-0.183691,0.0126756,-0.183691,-0.086861,0,0.242297,-0.572629,0.832758,0.572629,0.832758,0.572629,0.0631314,0.683943,-0.647626,-0.372322,0.647626,-0.372322,0.647626,-0.878269,-0.896306,-0.643406,-0.601699,0.643406,-0.601699,0.643406,0.728212;


    auto dls = A * A.transpose() + Eigen::MatrixXd::Identity(6, 6) * 1e-4;
    Eigen::LLT<Eigen::MatrixXd> llt(dls); // Perform Cholesky decomposition
    Eigen::MatrixXd L = llt.matrixL(); // Get the lower triangular factor
    Eigen::MatrixXd U = llt.matrixU(); // Get the upper triangular factor

    std::cout << "L:\n" << L << std::endl;
    std::cout << "U:\n" << U << std::endl;
    // You can also use llt.solve(b) to solve linear systems Ax = b

    auto decomposed = cholesky_factor<Eigen::MatrixXd, double>(dls);
    std::cout << "Decomposed:\n" << decomposed << std::endl;

    // verify that lower triangular matrix is same
    for (int i = 0; i < L.rows(); ++i) {
        for (int j = 0; j <= i; ++j) {
            if (std::abs(L(i, j) - decomposed(i, j)) > 1e-6) {
                throw std::runtime_error("Cholesky decomposition does not match!");
            }
        }
    }

    auto x = cholesky_solve<Eigen::MatrixXd, Eigen::VectorXd, double>(decomposed, b);
    std::cout << "Solution x:\n" << x.transpose() << std::endl;
    // Verify the solution
    auto y = llt.solve(b);
    std::cout << "Solution y (using Eigen's solver):\n" << y.transpose() << std::endl;

    // Verify both separately
    auto Ax = A.transpose() * x;
    std::cout << "Ax:\n" << Ax.transpose() << std::endl;
    std::cout << "b:\n" << b.transpose() << std::endl;

    auto Ay = A.transpose() * y;
    std::cout << "Ay:\n" << Ay.transpose() << std::endl;
    std::cout << "b:\n" << b.transpose() << std::endl;

    return 0;
}