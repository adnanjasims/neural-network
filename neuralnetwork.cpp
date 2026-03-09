#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <cassert>

using namespace std;

//class to handle 2d matrices and lin alg. operations
class Matrix {
public:
    int rows, cols;
    vector<vector<double>> data;
    Matrix(int r, int c) : rows(r), cols(c), data(r, vector<double>(c, 0.0)) {}

    //initializing matrix with random weights between -1.0 and 1.0
    void randomize() {
        random_device rd;
        mt19937 gen(rd());
        uniform_real_distribution<> dis(-1.0, 1.0);
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                data[i][j] = dis(gen);
            }
        }
    }

    //Matrix multiplication
    static Matrix multiply(const Matrix& a, const Matrix& b) {
        assert(a.cols == b.rows && "Matrix dimensions must match for multiplication.");
        Matrix result(a.rows, b.cols);
        for (int i = 0; i < a.rows; ++i) {
            for (int j = 0; j < b.cols; ++j) {
                double sum = 0.0;
                for (int k = 0; k < a.cols; ++k) {
                    sum += a.data[i][k] * b.data[k][j];
                }
                result.data[i][j] = sum;
            }
        }
        return result;
    }

    //Element-wise matrix addition (used for adding biases)
    void add(const Matrix& other) {
        assert(rows == other.rows && cols == other.cols && "Matrix dimensions must match for addition.");
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                data[i][j] += other.data[i][j];
            }
        }
    }
};

//Neural Network Class implementing a forward pass
class SimpleNN {
private:
    Matrix weights_input_hidden;
    Matrix bias_hidden;
    Matrix weights_hidden_output;
    Matrix bias_output;

    //ReLU Activation Function
    void applyReLU(Matrix& m) {
        for (int i = 0; i < m.rows; ++i) {
            for (int j = 0; j < m.cols; ++j) {
                m.data[i][j] = max(0.0, m.data[i][j]);
            }
        }
    }
    //Sigmoid Activation (for probabilities)
    void applySigmoid(Matrix& m) {
        for (int i = 0; i < m.rows; ++i) {
            for (int j = 0; j < m.cols; ++j) {
                m.data[i][j] = 1.0 / (1.0 + exp(-m.data[i][j]));
            }
        }
    }
public:
    //initializing the network architecture
    SimpleNN(int input_nodes, int hidden_nodes, int output_nodes) 
        : weights_input_hidden(hidden_nodes, input_nodes),
          bias_hidden(hidden_nodes, 1),
          weights_hidden_output(output_nodes, hidden_nodes),
          bias_output(output_nodes, 1) {
        
        weights_input_hidden.randomize();
        bias_hidden.randomize();
        weights_hidden_output.randomize();
        bias_output.randomize();
    }
    Matrix feedForward(Matrix input) {
        //Hidden Layer calculations: (Weights * Inputs) + Biases
        Matrix hidden = Matrix::multiply(weights_input_hidden, input);
        hidden.add(bias_hidden);
        applyReLU(hidden);

        //Output Layer calculations
        Matrix output = Matrix::multiply(weights_hidden_output, hidden);
        output.add(bias_output);
        applySigmoid(output);

        return output;
    }
};
int main() {
    cout << "Initializing neural network architecture..." << endl;
    
    //Create network: 3 inputs, 4 hidden nodes, 1 output
    SimpleNN nn(3, 4, 1);

    //Create dummy input matrix (3 rows, 1 column)
    Matrix inputs(3, 1);
    inputs.data[0][0] = 0.5;
    inputs.data[1][0] = 0.8;
    inputs.data[2][0] = 0.2;

    cout << "Performing forward pass..." << endl;
    Matrix output = nn.feedForward(inputs);

    cout << "Network Output Probability: " << output.data[0][0] << endl;

    return 0;
}
