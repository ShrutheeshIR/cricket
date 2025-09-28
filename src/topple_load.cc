
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
        std::cout << cell;
        row_data.push_back(ADCG(std::stod(cell)));
    }
    std::cout << " --> " << row_data.size() << std::endl;

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
        
      auto rdata = assign_line_to_matrix(file);
      ADMatrixXs layer_1_w =  Eigen::Map<ADMatrixXs>(rdata.data(), 6 * dof, 64);

      rdata = assign_line_to_matrix(file);
      ADMatrixXs layer_1_b =  Eigen::Map<ADMatrixXs>(rdata.data(), 1, 64);

      rdata = assign_line_to_matrix(file);
      ADMatrixXs layer_2_w =  Eigen::Map<ADMatrixXs>(rdata.data(), 64, 64);
      rdata = assign_line_to_matrix(file);
      ADMatrixXs layer_2_b =  Eigen::Map<ADMatrixXs>(rdata.data(), 1, 64);

      rdata = assign_line_to_matrix(file);
      ADMatrixXs layer_3_w =  Eigen::Map<ADMatrixXs>(rdata.data(), 64, 4 * dof + 1);
      rdata = assign_line_to_matrix(file);
      ADMatrixXs layer_3_b =  Eigen::Map<ADMatrixXs>(rdata.data(), 1, 4 * dof + 1);

      // read from file
      layer_weights.push_back(layer_1_w);
      layer_weights.push_back(layer_2_w);
      layer_weights.push_back(layer_3_w);

      layer_biases.push_back(layer_1_b);
      layer_biases.push_back(layer_2_b);
      layer_biases.push_back(layer_3_b);

      inp_size = 6 * dof;
      out_size = 4 * dof + 1;
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

auto trace_forward(
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

    auto x = ad_x.transpose();
    std::cout << x.rows() << ", " << x.cols() << std::endl;
    ADCG zero(0.0);

    for(auto i=0u; i < fnn_module.layer_weights.size(); i++)
    {
        std::cout << i << " " << x << std::endl;
        std::cout << fnn_module.layer_biases[i] << std::endl;
        x = x * fnn_module.layer_weights[i] + fnn_module.layer_biases[i];
        if (i < fnn_module.layer_weights.size() - 1) {
            for(auto j=0; j < x.size(); j++)
                x(j) = CondExpGe(x(j), zero, x(j), x(j) * 0.25);
        }
    }


    std::size_t n_out = fnn_module.out_size;

    ADVectorXs data(n_out);

    for (auto i=0U; i < n_out; i++)
        data[i] = x(i);
    

    // trace_frame(info.end_effector_index, ad_data, data, n_spheres_data + n_bounding_spheres_data);

    // Create the AD function
    ADFun<CGD> topple_nn(ad_x, data);

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
