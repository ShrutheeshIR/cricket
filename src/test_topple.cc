#include <cppad/cg/support/cppadcg_eigen.hpp>

#include <fmt/core.h>
#include <nlohmann/json.hpp>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "lang_gen.hh"

using namespace CppAD;
using namespace CppAD::cg;

using CGD = double;
using ADCG = AD<CGD>;

using ADVectorXs = Eigen::Matrix<ADCG, Eigen::Dynamic, 1>;
using ADMatrixXs = Eigen::Matrix<ADCG, Eigen::Dynamic, Eigen::Dynamic>;

std::vector<ADCG> assign_line_to_matrix(std::ifstream &file)
{
    std::string line;
    std::getline(file, line);

    std::stringstream ss(line);
    std::string cell;

    std::vector<ADCG> row_data;
    while (std::getline(ss, cell, ','))
    {
        row_data.push_back(ADCG(std::stod(cell)));
    }

    return row_data;
}

struct ToppleNN
{
    ToppleNN(const std::filesystem::path &weights_path, const size_t dof)
    {
        std::ifstream file(weights_path);
        if (not file)
        {
            throw std::runtime_error(fmt::format("Failed to open weights file {}", weights_path.string()));
        }

        const int hsize = 1024;
        const int deg = 7;

        auto rdata = assign_line_to_matrix(file);
        ADMatrixXs inp_layer = Eigen::Map<ADMatrixXs>(rdata.data(), 6 * dof, hsize);
        rdata = assign_line_to_matrix(file);
        ADMatrixXs inp_b = Eigen::Map<ADMatrixXs>(rdata.data(), 1, hsize);

        rdata = assign_line_to_matrix(file);
        ADMatrixXs norm_0_w = Eigen::Map<ADMatrixXs>(rdata.data(), 1, hsize);
        rdata = assign_line_to_matrix(file);
        ADMatrixXs norm_0_b = Eigen::Map<ADMatrixXs>(rdata.data(), 1, hsize);

        rdata = assign_line_to_matrix(file);
        ADMatrixXs linear_1_w = Eigen::Map<ADMatrixXs>(rdata.data(), hsize, hsize);
        rdata = assign_line_to_matrix(file);
        ADMatrixXs linear_1_b = Eigen::Map<ADMatrixXs>(rdata.data(), 1, hsize);

        rdata = assign_line_to_matrix(file);
        ADMatrixXs norm_1_w = Eigen::Map<ADMatrixXs>(rdata.data(), 1, hsize);
        rdata = assign_line_to_matrix(file);
        ADMatrixXs norm_1_b = Eigen::Map<ADMatrixXs>(rdata.data(), 1, hsize);

        rdata = assign_line_to_matrix(file);
        ADMatrixXs linear_2_w = Eigen::Map<ADMatrixXs>(rdata.data(), hsize, hsize);
        rdata = assign_line_to_matrix(file);
        ADMatrixXs linear_2_b = Eigen::Map<ADMatrixXs>(rdata.data(), 1, hsize);

        rdata = assign_line_to_matrix(file);
        ADMatrixXs norm_2_w = Eigen::Map<ADMatrixXs>(rdata.data(), 1, hsize);
        rdata = assign_line_to_matrix(file);
        ADMatrixXs norm_2_b = Eigen::Map<ADMatrixXs>(rdata.data(), 1, hsize);

        rdata = assign_line_to_matrix(file);
        ADMatrixXs linear_3_w = Eigen::Map<ADMatrixXs>(rdata.data(), hsize, hsize);
        rdata = assign_line_to_matrix(file);
        ADMatrixXs linear_3_b = Eigen::Map<ADMatrixXs>(rdata.data(), 1, hsize);

        rdata = assign_line_to_matrix(file);
        ADMatrixXs norm_3_w = Eigen::Map<ADMatrixXs>(rdata.data(), 1, hsize);
        rdata = assign_line_to_matrix(file);
        ADMatrixXs norm_3_b = Eigen::Map<ADMatrixXs>(rdata.data(), 1, hsize);

        rdata = assign_line_to_matrix(file);
        ADMatrixXs linear_4_w = Eigen::Map<ADMatrixXs>(rdata.data(), hsize, hsize);
        rdata = assign_line_to_matrix(file);
        ADMatrixXs linear_4_b = Eigen::Map<ADMatrixXs>(rdata.data(), 1, hsize);

        rdata = assign_line_to_matrix(file);
        ADMatrixXs norm_4_w = Eigen::Map<ADMatrixXs>(rdata.data(), 1, hsize);
        rdata = assign_line_to_matrix(file);
        ADMatrixXs norm_4_b = Eigen::Map<ADMatrixXs>(rdata.data(), 1, hsize);

        rdata = assign_line_to_matrix(file);
        ADMatrixXs linear_5_w = Eigen::Map<ADMatrixXs>(rdata.data(), hsize, hsize);
        rdata = assign_line_to_matrix(file);
        ADMatrixXs linear_5_b = Eigen::Map<ADMatrixXs>(rdata.data(), 1, hsize);

        rdata = assign_line_to_matrix(file);
        ADMatrixXs norm_5_w = Eigen::Map<ADMatrixXs>(rdata.data(), 1, hsize);
        rdata = assign_line_to_matrix(file);
        ADMatrixXs norm_5_b = Eigen::Map<ADMatrixXs>(rdata.data(), 1, hsize);

        rdata = assign_line_to_matrix(file);
        ADMatrixXs out_layer = Eigen::Map<ADMatrixXs>(rdata.data(), hsize, dof * (deg - 1) + 1);
        rdata = assign_line_to_matrix(file);
        ADMatrixXs out_b = Eigen::Map<ADMatrixXs>(rdata.data(), 1, dof * (deg - 1) + 1);

        layer_weights.push_back(inp_layer);
        layer_weights.push_back(norm_0_w);
        layer_weights.push_back(linear_1_w);
        layer_weights.push_back(norm_1_w);
        layer_weights.push_back(linear_2_w);
        layer_weights.push_back(norm_2_w);
        layer_weights.push_back(linear_3_w);
        layer_weights.push_back(norm_3_w);
        layer_weights.push_back(linear_4_w);
        layer_weights.push_back(norm_4_w);
        layer_weights.push_back(linear_5_w);
        layer_weights.push_back(norm_5_w);
        layer_weights.push_back(out_layer);

        layer_biases.push_back(inp_b);
        layer_biases.push_back(norm_0_b);
        layer_biases.push_back(linear_1_b);
        layer_biases.push_back(norm_1_b);
        layer_biases.push_back(linear_2_b);
        layer_biases.push_back(norm_2_b);
        layer_biases.push_back(linear_3_b);
        layer_biases.push_back(norm_3_b);
        layer_biases.push_back(linear_4_b);
        layer_biases.push_back(norm_4_b);
        layer_biases.push_back(linear_5_b);
        layer_biases.push_back(norm_5_b);
        layer_biases.push_back(out_b);


        inp_size = 6 * dof;
        out_size = dof * (deg - 1) + 1;
    }

    std::vector<ADMatrixXs> layer_weights;
    std::vector<ADMatrixXs> layer_biases;
    size_t inp_size;
    size_t out_size;
};

struct Traced
{
    std::string code;
    std::size_t temp_variables;
    std::size_t outputs;
};

auto layer_norm(ADVectorXs z)
{
    const ADCG eps(1e-5);
    auto mean = z.mean();
    auto variance = (z.array() - mean).square().mean();
    auto normalized = (z.array() - mean) / CppAD::sqrt(variance + eps);
    return normalized.matrix();
}

auto trace_forward(
    const ToppleNN &fnn_module,
    ADVectorXs ad_x
) -> CppAD::vector<CGD>
{

    const size_t num_inp = fnn_module.inp_size;
    Independent(ad_x);

    const int hsize = 1024;
    const int dof = 7;
    const int deg = 7;

    auto x = ad_x.transpose();
    ADCG zero(0.0);

    ADVectorXs ad_l1(hsize);
    auto l1 = ad_l1.transpose();

    ADVectorXs ad_l2(hsize);
    auto l2 = ad_l2.transpose();

    ADVectorXs ad_l3(hsize);
    auto l3 = ad_l3.transpose();

    ADVectorXs ad_l4(hsize);
    auto l4 = ad_l4.transpose();

    ADVectorXs ad_l5(hsize);
    auto l5 = ad_l5.transpose();

    ADVectorXs ad_y(dof * (deg - 1) + 1);
    auto y = ad_y.transpose();

    std::vector<ADMatrixXs> layers;
    layers.push_back(x);
    layers.push_back(l1);
    layers.push_back(l2);
    layers.push_back(l3);
    layers.push_back(l4);
    layers.push_back(l5);

    std::cout << "Starting forward pass" << std::endl;
    for (auto i = 1; i < layers.size(); i++)
    {
        auto residual = layers[i - 1];
        layers[i] = residual * fnn_module.layer_weights[2 * (i - 1)] + fnn_module.layer_biases[2 * (i - 1)];
        layers[i] = layer_norm(layers[i]) * fnn_module.layer_weights[2 * (i - 1) + 1] + fnn_module.layer_biases[2 * (i - 1) + 1];
        for(auto j=0; j < layers[i].size(); j++) {
            layers[i](j) = CondExpGe(layers[i](j), zero, layers[i](j), zero);
        }
        layers[i] = residual + layers[i];
    }
    y = layers.back() * fnn_module.layer_weights.back() + fnn_module.layer_biases.back();
    std::cout << "Finished forward pass" << std::endl;

    std::size_t n_out = fnn_module.out_size;
    ADVectorXs data(n_out);

    for (auto i = 0U; i < n_out; i++)
        data[i] = y(i);

    ADFun<CGD> topple_nn(ad_x, data);

    CppAD::vector<CGD> ind_vars(num_inp);
    for (auto i=0U; i < num_inp; i++)
        ind_vars[i] = CppAD::Value(ad_inp[i]);

    CppAD::vector<CGD> result = topple_nn.Forward(0, ind_vars);

    return result;

    // LanguageCCustom<double> langC("double");
    // LangCDefaultVariableNameGenerator<double> nameGen;

    // std::ostringstream function_code;
    // handler.generateCode(function_code, langC, result, nameGen);

    // return Traced{function_code.str(), handler.getTemporaryVariableCount(), n_out};
}

int main()
{
    try
    {
        const std::filesystem::path config_path = std::filesystem::path(TOPPLE_CRICKET_SOURCE_DIR) / "resources" / "topple.json";
        std::ifstream config_file(config_path);
        if (not config_file)
        {
            throw std::runtime_error("Failed to open topple.json test config");
        }

        nlohmann::json data = nlohmann::json::parse(config_file);
        const std::filesystem::path parent_path = config_path.parent_path();

        if (not data.contains("dof"))
        {
            throw std::runtime_error("topple.json is missing dof");
        }

        ToppleNN fnn_module(parent_path / data["weights"].get<std::string>(), data["dof"].get<std::size_t>());

        ADVectorXs ad_x = {
            -0.3709,  0.8812,  0.2831,  0.8987, -0.6514,  1.3344, -1.0018, -0.1213,
         -1.0319, -0.6773,  1.3641, -0.3000, -0.3132,  0.3744,  1.9019, -2.0400,
          0.8815, -0.9878, -2.0240, -0.8209,  1.1444,  1.2264,  2.3361,  1.6711,
         -0.6064,  1.8608, -0.5152,  0.7335, -0.0483,  0.0146,  0.3823, -1.5024,
         -0.1906, -0.1035,  1.0389,  1.5204,  1.3464, -1.2553, -1.6046,  1.0720,
         -0.4290,  1.6618
        };
        // for (auto i = 0U; i < fnn_module.inp_size; ++i)
        //     ad_x[i] = ADCG(0.0);

        const auto result = trace_forward(fnn_module, ad_x);
        std::cout << "Result of trace_forward:\n";
        for (auto i=0U; i < result.size(); i++)            std::cout << result[i] << " ";
        std::cout << std::endl;


        std::cout << "trace_forward test passed\n";
        return 0;
    }
    catch (const std::exception &e)
    {
        std::cerr << "test_topple failed: " << e.what() << '\n';
        return 1;
    }
}