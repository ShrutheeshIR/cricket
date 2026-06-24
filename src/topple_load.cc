
#include <fmt/core.h>
#include <inja/inja.hpp>
#include <cxxopts.hpp>

#include <filesystem>
#include <stdexcept>
#include <vector>
#include <optional>
#include <nlohmann/json.hpp>

#include <fstream>  // For file stream operations (e.g., std::ifstream)
#include <string>   // For string manipulation (e.g., std::string)
#include <sstream>  // For string stream operations (e.g., std::stringstream)

#include <cmath>
#include <boost/mpl/int.hpp>
#include <cppad/cg/support/cppadcg_eigen.hpp>

#include "lang_gen.hh"

using namespace CppAD;
using namespace CppAD::cg;

// Typedef for AD types
using CGD = CG<double>;
using ADCG = AD<CGD>;

using ADVectorXs = Eigen::Matrix<ADCG, Eigen::Dynamic, 1>;
using ADMatrixXs = Eigen::Matrix<ADCG, Eigen::Dynamic, Eigen::Dynamic>;

std::vector<ADCG> assign_line_to_matrix(std::ifstream& file) {
    std::string line;
    std::getline(file, line);

    std::stringstream ss(line);
    std::string cell;

    std::vector<ADCG> row_data;
    while (std::getline(ss, cell, ',')){
        // std::cout << cell;
        row_data.push_back(ADCG(std::stod(cell)));
    }
    // std::cout << " --> " << row_data.size() << std::endl;

    return row_data;
}

struct ToppleNN
{
    ToppleNN(
        const std::filesystem::path &weights_path,
        const size_t dof
  )
    {
      std::cout << "Starting construction of NN" << std::endl;
      std::ifstream file(weights_path);

      // each matrix is stored in a single line
    
      const int hsize = 1024;
      const int deg = 7;

      std::cout << "Read layers from file" << std::endl;
      auto rdata = assign_line_to_matrix(file);
      ADMatrixXs inp_layer =  Eigen::Map<ADMatrixXs>(rdata.data(), 6 * dof, hsize);
      rdata = assign_line_to_matrix(file);
      ADMatrixXs inp_b =  Eigen::Map<ADMatrixXs>(rdata.data(), 1, hsize);

      rdata = assign_line_to_matrix(file);
      ADMatrixXs norm_0_w =  Eigen::Map<ADMatrixXs>(rdata.data(), 1, hsize);
      rdata = assign_line_to_matrix(file);
      ADMatrixXs norm_0_b =  Eigen::Map<ADMatrixXs>(rdata.data(), 1, hsize);

      rdata = assign_line_to_matrix(file);
      ADMatrixXs linear_1_w =  Eigen::Map<ADMatrixXs>(rdata.data(), hsize, hsize);
      rdata = assign_line_to_matrix(file);
      ADMatrixXs linear_1_b =  Eigen::Map<ADMatrixXs>(rdata.data(), 1, hsize);

      rdata = assign_line_to_matrix(file);
      ADMatrixXs norm_1_w =  Eigen::Map<ADMatrixXs>(rdata.data(), 1, hsize);
      rdata = assign_line_to_matrix(file);
      ADMatrixXs norm_1_b =  Eigen::Map<ADMatrixXs>(rdata.data(), 1, hsize);

      rdata = assign_line_to_matrix(file);
      ADMatrixXs linear_2_w =  Eigen::Map<ADMatrixXs>(rdata.data(), hsize, hsize);
      rdata = assign_line_to_matrix(file);
      ADMatrixXs linear_2_b =  Eigen::Map<ADMatrixXs>(rdata.data(), 1, hsize);

      rdata = assign_line_to_matrix(file);
      ADMatrixXs norm_2_w =  Eigen::Map<ADMatrixXs>(rdata.data(), 1, hsize);
      rdata = assign_line_to_matrix(file);
      ADMatrixXs norm_2_b =  Eigen::Map<ADMatrixXs>(rdata.data(), 1, hsize);

      rdata = assign_line_to_matrix(file);
      ADMatrixXs linear_3_w =  Eigen::Map<ADMatrixXs>(rdata.data(), hsize, hsize);
      rdata = assign_line_to_matrix(file);
      ADMatrixXs linear_3_b =  Eigen::Map<ADMatrixXs>(rdata.data(), 1, hsize);

      rdata = assign_line_to_matrix(file);
      ADMatrixXs norm_3_w =  Eigen::Map<ADMatrixXs>(rdata.data(), 1, hsize);
      rdata = assign_line_to_matrix(file);
      ADMatrixXs norm_3_b =  Eigen::Map<ADMatrixXs>(rdata.data(), 1, hsize);

      rdata = assign_line_to_matrix(file);
      ADMatrixXs linear_4_w =  Eigen::Map<ADMatrixXs>(rdata.data(), hsize, hsize);
      rdata = assign_line_to_matrix(file);
      ADMatrixXs linear_4_b =  Eigen::Map<ADMatrixXs>(rdata.data(), 1, hsize);

      rdata = assign_line_to_matrix(file);
      ADMatrixXs norm_4_w =  Eigen::Map<ADMatrixXs>(rdata.data(), 1, hsize);
      rdata = assign_line_to_matrix(file);
      ADMatrixXs norm_4_b =  Eigen::Map<ADMatrixXs>(rdata.data(), 1, hsize);

      rdata = assign_line_to_matrix(file);
      ADMatrixXs linear_5_w =  Eigen::Map<ADMatrixXs>(rdata.data(), hsize, hsize);
      rdata = assign_line_to_matrix(file);
      ADMatrixXs linear_5_b =  Eigen::Map<ADMatrixXs>(rdata.data(), 1, hsize);

      rdata = assign_line_to_matrix(file);
      ADMatrixXs norm_5_w =  Eigen::Map<ADMatrixXs>(rdata.data(), 1, hsize);
      rdata = assign_line_to_matrix(file);
      ADMatrixXs norm_5_b =  Eigen::Map<ADMatrixXs>(rdata.data(), 1, hsize);

      rdata = assign_line_to_matrix(file);
      ADMatrixXs out_layer =  Eigen::Map<ADMatrixXs>(rdata.data(), hsize, dof * (deg - 1) + 1);
      rdata = assign_line_to_matrix(file);
      ADMatrixXs out_b =  Eigen::Map<ADMatrixXs>(rdata.data(), 1, dof * (deg - 1) + 1);
    // std::cout << "reached" << std::endl;

      // read from file
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
      std::cout << "Completed construction with " << dof << std::endl;

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

auto layer_norm(ADVectorXs z) {
    const float eps = 0.00001;
    auto mean = z.mean();
    auto variance = (z.array() - mean).square().mean();
    auto normalized = (z.array() - mean) / CppAD::sqrt(variance + eps);
    return normalized.matrix();
}

auto trace_forward(
    const ToppleNN &fnn_module
)
{
    // Inputs are assumed normalized to zscore
    ADVectorXs ad_x(fnn_module.inp_size);
    for (auto i = 0U; i < fnn_module.inp_size; ++i)
        ad_x[i] = ADCG(0.0);

    Independent(ad_x);

    // auto x = ad_x;
    // ADVectorXs ix(fnn_module.inp_size);
    // for (auto i = 0U; i < fnn_module.inp_size; ++i)
    //     ix[i] = ad_x[i];

    const int hsize = 1024;
    const int dof = 7;
    const int deg = 7;

    // Excuse this disgustingness, i really just dont want to think about this right now
    auto x = ad_x.transpose();
    std::cout << x.rows() << ", " << x.cols() << std::endl;
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
    // layers.push_back(n1);
    layers.push_back(l2);
    // layers.push_back(n2);
    layers.push_back(l3);
    // layers.push_back(n3);
    layers.push_back(l4);
    // layers.push_back(n4);
    layers.push_back(l5);
    // layers.push_back(n5);

    // forward pass
    
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

    // Perform the forward pass (Linear, layernorm, relu)
    // 1. Linear
    // l1 = x * fnn_module.layer_weights[0] + fnn_module.layer_biases[0];
    // // 2. Layernorm
    // n1 = layer_norm(l1) * fnn_module.layer_weights[1] + fnn_module.layer_biases[1];
    // // 3. ReLU
    // for(auto i=0; i < n1.size(); i++)
    //     n1(i) = CondExpGe(n1(i), zero, n1(i), zero);
    
    // l2 = n1 * fnn_module.layer_weights[2] + fnn_module.layer_biases[2];
    // // 2. Layernorm
    // n2 = layer_norm(l2) * fnn_module.layer_weights[3] + fnn_module.layer_biases[3];
    // // 3. ReLU
    // for(auto i=0; i < n2.size(); i++)
    //     n2(i) = CondExpGe(n2(i), zero, n2(i), zero);

    // l3 = n1 * fnn_module.layer_weights[4] + fnn_module.layer_biases[4];
    // // 2. Layernorm
    // n3 = layer_norm(l3) * fnn_module.layer_weights[5] + fnn_module.layer_biases[5];
    // for(auto i=0; i < n3.size(); i++)
    //     n3(i) = CondExpGe(n3(i), zero, n3(i), zero);

    // l4 = n3 * fnn_module.layer_weights[4] + fnn_module.layer_biases[4];
    // // 2. Layernorm
    // n4 = layer_norm(l4) * fnn_module.layer_weights[5] + fnn_module.layer_biases[5];

    // // 3. ReLU
    // for(auto i=0; i < n1.size(); i++)
    //     n2(i) = CondExpGe(n2(i), zero, n2(i), zero);

    // y = h1 * fnn_module.layer_weights[1] + fnn_module.layer_biases[1];

    // for(auto i=0u; i < fnn_module.layer_weights.size(); i++)
    // {
    //     std::cout << i << " " << x << std::endl;
    //     std::cout << fnn_module.layer_biases[i] << std::endl;
    //     x = x * fnn_module.layer_weights[i] + fnn_module.layer_biases[i];
    //     if (i < fnn_module.layer_weights.size() - 1) {
    //         for(auto j=0; j < x.size(); j++)
    //             x(j) = CondExpGe(x(j), zero, x(j), zero);
    //     }
    // }


    std::size_t n_out = fnn_module.out_size;
    std::cout << n_out << std::endl;
    std::cout << y.size() << std::endl;
    ADVectorXs data(n_out);

    for (auto i=0U; i < n_out; i++)
        data[i] = y(i);

    
    // trace_frame(info.end_effector_index, ad_data, data, n_spheres_data + n_bounding_spheres_data);

    // Create the AD function
    ADFun<CGD> topple_nn(ad_x, data); // seg fault?

    CodeHandler<double> handler;
    CppAD::vector<CGD> ind_vars(fnn_module.inp_size);
    handler.makeVariables(ind_vars);

    CppAD::vector<CGD> result = topple_nn.Forward(0, ind_vars);

    LanguageCCustom<double> langC("double");
    LangCDefaultVariableNameGenerator<double> nameGen;

    std::ostringstream function_code;
    handler.generateCode(function_code, langC, result, nameGen);
    
    return Traced{function_code.str(), handler.getTemporaryVariableCount(), n_out};
}

auto trace_forward_and_backward(
    const ToppleNN &fnn_module
)
{

    ADVectorXs ad_x(fnn_module.inp_size);
    for (auto i = 0U; i < fnn_module.inp_size; ++i)
        ad_x[i] = ADCG(0.0);

    Independent(ad_x);

    // auto x = ad_x;
    // ADVectorXs ix(fnn_module.inp_size);
    // for (auto i = 0U; i < fnn_module.inp_size; ++i)
    //     ix[i] = ad_x[i];

    // excuse this disgustingness
    auto x = ad_x.transpose();
    std::cout << x.rows() << ", " << x.cols() << std::endl;
    ADCG zero(0.0);

    ADVectorXs ad_h1(32);
    auto h1 = ad_h1.transpose();

    ADVectorXs ad_y(36);
    auto y = ad_y.transpose();

    h1 = x * fnn_module.layer_weights[0] + fnn_module.layer_biases[0];
    for(auto i=0; i < h1.size(); i++)
        h1(i) = CondExpGe(h1(i), zero, h1(i), zero);

    y = h1 * fnn_module.layer_weights[1] + fnn_module.layer_biases[1];

    // for(auto i=0u; i < fnn_module.layer_weights.size(); i++)
    // {
    //     std::cout << i << " " << x << std::endl;
    //     std::cout << fnn_module.layer_biases[i] << std::endl;
    //     x = x * fnn_module.layer_weights[i] + fnn_module.layer_biases[i];
    //     if (i < fnn_module.layer_weights.size() - 1) {
    //         for(auto j=0; j < x.size(); j++)
    //             x(j) = CondExpGe(x(j), zero, x(j), zero);
    //     }
    // }


    std::size_t n_out = 1; // we care only about time, the last entry
    std::cout << n_out << std::endl;
    std::cout << y.size() << std::endl;
    ADVectorXs data(n_out);

    data[0] = y(fnn_module.out_size - 1); // time is the last entry

    // for (auto i=0U; i < n_out; i++)
    //     data[i] = y(i);

    
    // trace_frame(info.end_effector_index, ad_data, data, n_spheres_data + n_bounding_spheres_data);

    // Create the AD function
    ADFun<CGD> topple_nn(ad_x, data); // seg fault?

    CodeHandler<double> handler;
    CppAD::vector<CGD> ind_vars(fnn_module.inp_size);
    handler.makeVariables(ind_vars);

    CppAD::vector<CGD> result = topple_nn.Forward(0, ind_vars);
    CppAD::vector<CGD> jac = topple_nn.Jacobian(ind_vars);

    // insert jac at the end of result flattened
    std::move(jac.begin(), jac.end(), std::back_inserter(result));

    const size_t n_result = result.size();

    LanguageCCustom<double> langC("double");
    LangCDefaultVariableNameGenerator<double> nameGen;

    std::ostringstream function_code;
    handler.generateCode(function_code, langC, result, nameGen);
    
    return Traced{function_code.str(), handler.getTemporaryVariableCount(), n_result};
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


    // std::optional<std::string> end_effector_name = {};
    size_t dof;
    if (data.contains("dof"))
    {
        dof = data["dof"];
    }

    ToppleNN fnn_module(parent_path / data["weights"], dof);

    auto traced_forward_code = trace_forward(fnn_module);

    data["forward_code"] = traced_forward_code.code;
    data["forward_code_vars"] = traced_forward_code.temp_variables;
    data["forward_code_output"] = traced_forward_code.outputs;

    auto traced_forward_and_backward_code = trace_forward_and_backward(fnn_module);
    data["forward_and_backward_code"] = traced_forward_and_backward_code.code;
    data["forward_and_backward_code_vars"] = traced_forward_and_backward_code.temp_variables;
    data["forward_and_backward_code_output"] = traced_forward_and_backward_code.outputs;

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
