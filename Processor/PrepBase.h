/*
 * SubDataFilesBase.h
 *
 */

#ifndef PROCESSOR_PREPBASE_H_
#define PROCESSOR_PREPBASE_H_

#include <string>
using namespace std;

#include "Math/field_types.h"
#include "Tools/matmul.h"
#include "Tools/conv2d.h"

class PrepBase
{
public:
    static string get_suffix(int thread_num);

    static string get_filename(const string& prep_data_dir, Dtype type,
            const string& type_short, int my_num, int thread_num = 0);
    static string get_input_filename(const string& prep_data_dir,
            const string& type_short, int input_player, int my_num,
            int thread_num = 0);
    static string get_edabit_filename(const string& prep_data_dir, int n_bits,
            int my_num, int thread_num = 0);
    static string get_matmul_filename(const string& prep_data_dir, matmul_dimensions dimensions, const string& type_short, int my_num, int thread_num = 0);
    static string get_conv2d_filename(const string& prep_data_dir, convolution_dimensions dimensions, const string& type_short, int my_num, int thread_num = 0);
    static string get_conv2d_filename(const string& prep_data_dir, depthwise_convolution_triple_dimensions dimensions, const string& type_short, int my_num, int thread_num = 0);

    static void print_left(const char* name, size_t n,
            const string& type_string, size_t used);
    static void print_left_edabits(size_t n, size_t n_batch, bool strict,
            int n_bits, size_t used);
};

string get_matmul_file_prefix(const string& prep_data_dir, matmul_dimensions dimensions, const string& type_short);
string get_conv2d_file_prefix(const string& prep_data_dir, convolution_dimensions dimensions, const string& type_short);
string get_conv2d_file_prefix(const string& prep_data_dir, depthwise_convolution_triple_dimensions dimensions, const string& type_short);

#endif /* PROCESSOR_PREPBASE_H_ */
