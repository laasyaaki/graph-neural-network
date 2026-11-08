#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <iostream>

namespace py = pybind11;


void softmax_regression_epoch_cpp(const float *X, const unsigned char *y,
								  float *theta, size_t m, size_t n, size_t k,
								  float lr, size_t batch)
{
    /**
     * A C++ version of the softmax regression epoch code.  This should run a
     * single epoch over the data defined by X and y (and sizes m,n,k), and
     * modify theta in place.  Your function will probably want to allocate
     * (and then delete) some helper arrays to store the logits and gradients.
     *
     * Args:
     *     X (const float *): pointer to X data, of size m*n, stored in row
     *          major (C) format
     *     y (const unsigned char *): pointer to y data, of size m
     *     theta (float *): pointer to theta data, of size n*k, stored in row
     *          major (C) format
     *     m (size_t): number of examples
     *     n (size_t): input dimension
     *     k (size_t): number of classes
     *     lr (float): learning rate / SGD step size
     *     batch (int): SGD minibatch size
     *
     * Returns:
     *     (None)
     */

    /// BEGIN YOUR CODE
    for (size_t start = 0; start < m; start += batch) {
        size_t batch_size = std::min(batch, m - start);

        float *probs = new float[batch_size * k];

        float *grad = new float[n * k]();

        for (size_t i = 0; i < batch_size; i++) {
            for (size_t c = 0; c < k; c++) {

                float sum = 0.0f;

                for (size_t j = 0; j < n; j++) {
                    sum += X[(start + i) * n + j] *
                           theta[j * k + c];
                }

                probs[i * k + c] = sum;
            }
        }

        for (size_t i = 0; i < batch_size; i++) {

            float max_logit = probs[i * k];

            for (size_t c = 1; c < k; c++) {
                if (probs[i * k + c] > max_logit) {
                    max_logit = probs[i * k + c];
                }
            }

            float denominator = 0.0f;

            for (size_t c = 0; c < k; c++) {
                probs[i * k + c] =
                    std::exp(probs[i * k + c] - max_logit);

                denominator += probs[i * k + c];
            }

            for (size_t c = 0; c < k; c++) {
                probs[i * k + c] /= denominator;
            }
        }

        for (size_t i = 0; i < batch_size; i++) {
            probs[i * k + y[start + i]] -= 1.0f;
        }

        for (size_t j = 0; j < n; j++) {
            for (size_t c = 0; c < k; c++) {

                float sum = 0.0f;

                for (size_t i = 0; i < batch_size; i++) {
                    sum += X[(start + i) * n + j] *
                           probs[i * k + c];
                }

                grad[j * k + c] = sum / batch_size;
            }
        }

        for (size_t j = 0; j < n; j++) {
            for (size_t c = 0; c < k; c++) {
                theta[j * k + c] -= lr * grad[j * k + c];
            }
        }

        delete[] probs;
        delete[] grad;
    }
    /// END YOUR CODE
}


/**
 * This is the pybind11 code that wraps the function above.  It's only role is
 * wrap the function above in a Python module, and you do not need to make any
 * edits to the code
 */
PYBIND11_MODULE(simple_ml_ext, m) {
    m.def("softmax_regression_epoch_cpp",
    	[](py::array_t<float, py::array::c_style> X,
           py::array_t<unsigned char, py::array::c_style> y,
           py::array_t<float, py::array::c_style> theta,
           float lr,
           int batch) {
        softmax_regression_epoch_cpp(
        	static_cast<const float*>(X.request().ptr),
            static_cast<const unsigned char*>(y.request().ptr),
            static_cast<float*>(theta.request().ptr),
            X.request().shape[0],
            X.request().shape[1],
            theta.request().shape[1],
            lr,
            batch
           );
    },
    py::arg("X"), py::arg("y"), py::arg("theta"),
    py::arg("lr"), py::arg("batch"));
}
