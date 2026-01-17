#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <cstring>
#include <chrono>

#include "mkl.h"
#include "generate_matrices.h"
#include "mkl_complex16.h"
#include "lindblad_utils.h"

using namespace std::chrono;

MKL_Complex16 Trace(const MKL_Complex16* matrix, int n) {
    MKL_Complex16 tr = {0.0, 0.0};
    for (int i = 0; i < n; ++i) {
        int diag_idx = i * n + i;
        tr.real += matrix[diag_idx].real;
        tr.imag += matrix[diag_idx].imag;
    }
    return tr;
}

// здесь и далее попробуем решить систему для константного H
// тогда матрица Q тоже будет константной

// Вспомогательная функция для печати вектора
void print_vector(const char* name, double t, const double* v, int M) {
    std::cout << name << "(t=" << t << "): [ ";
    for (int i = 0; i < M; ++i) {
        std::cout << v[i] << ", ";
    }
    std::cout << "]" << std::endl;
}

// Реализация нашей векторной функции f(t, v) = (Q(t) + R)v + K
// result = (Q(t) + R)v + K
void calculate_f(double t, const double* v, double* result, const double* Q, const double* R,
                 const double* K, double* workspace_A, int M) {
    double* A = workspace_A;

    // 1. Вычисляем матрицу A(t) = Q(t) + R
    double alpha = 1.0;  // Коэффициент для Q
    double beta = 1.0;   // Коэффициент для R
    mkl_domatadd('R',    // 'R' для RowMajor (C-стиль), 'C' для ColMajor (Fortran-стиль)
                 'N',    // 'N' (No-transpose) для Q
                 'N',    // 'N' (No-transpose) для R
                 M,      // Количество строк
                 M,      // Количество столбцов
                 alpha,  // Скаляр alpha (1.0)
                 Q,      // Матрица Q
                 M,      // Ведущий размер (lda) для Q
                 beta,   // Скаляр beta (1.0)
                 R,      // Матрица R
                 M,      // Ведущий размер (ldb) для R
                 A,      // Результирующая матрица A
                 M       // Ведущий размер (ldc) для A
    );

    // 2. Вычисляем A(t) * v
    // cblas_dgemv: result = alpha*A*v + beta*result
    // Мы хотим result = 1.0 * A * v + 0.0 * result
    beta = 0.0;
    cblas_dgemv(CblasRowMajor, CblasNoTrans, M, M,  // Размеры матрицы A
                alpha,                              // alpha
                A, M,                               // Матрица A и ее lda
                v, 1,                               // Вектор v и его инкремент
                beta,                               // beta
                result, 1);                         // Результирующий вектор и его инкремент

    // 3. Добавляем вектор K
    // cblas_daxpy: result = 1.0 * K + result
    cblas_daxpy(M, alpha, K, 1, result, 1);
}

std::vector<double> GetHCoef(MKL_Complex16* hamiltonian, int N) {
    std::vector<double> h_coeff;

    for (int j = 0; j < N; ++j) {
        for (int k = j + 1; k < N; ++k) {
            int index = j * N + k;
            h_coeff.push_back(sqrt(2.) * hamiltonian[index].real);
        }
    }

    for (int j = 0; j < N; ++j) {
        for (int k = j + 1; k < N; ++k) {
            int index = k * N + j;
            h_coeff.push_back(sqrt(2.) * hamiltonian[index].imag);
        }
    }

    for (int l = 0; l < N - 1; ++l) {
        double coeff = 0.;

        for (int k = 0; k < l + 1; ++k) {
            int index = k * N + k;
            coeff += hamiltonian[index].real / sqrt((l + 1) * (l + 2));
        }

        int index = (l + 1) * N + (l + 1);
        coeff += -sqrt(l + 1) * hamiltonian[index].real / sqrt(l + 2);

        h_coeff.push_back(coeff);
    }

    return h_coeff;
}

std::vector<MKL_Complex16> GetLCoef(MKL_Complex16* lindbladian, int N) {
    std::vector<MKL_Complex16> l_coeff;
    for (int j = 0; j < N; ++j) {
        for (int k = j + 1; k < N; ++k) {
            MKL_Complex16 coeff = (lindbladian[j * N + k] + lindbladian[k * N + j]) / sqrt(2);
            l_coeff.push_back(coeff);
        }
    }

    for (int j = 0; j < N; ++j) {
        for (int k = j + 1; k < N; ++k) {

            int ind_one = j * N + k;
            int ind_two = k * N + j;
            MKL_Complex16 coeff;
            coeff.real = lindbladian[ind_two].imag - lindbladian[ind_one].imag;
            coeff.imag = lindbladian[ind_one].real - lindbladian[ind_two].real;
            l_coeff.push_back(coeff / sqrt(2));
        }
    }

    for (int l = 0; l < N - 1; ++l) {
        MKL_Complex16 coeff = {0., 0.};

        for (int k = 0; k < l + 1; ++k) {
            int index = k * N + k;
            coeff += lindbladian[index] / sqrt((l + 1) * (l + 2));
        }

        int index = (l + 1) * N + (l + 1);
        coeff += -sqrt(l + 1) * lindbladian[index] / sqrt(l + 2);

        l_coeff.push_back(coeff);
    }

    return l_coeff;
}

int main() {
    // Параметры
    int N = 70;
    // int M = N * N - 1;

    // Cоздаем гамильтониан
    MKL_Complex16* hamiltonian;
    GenerateTracelessHamiltonian(N, 2, hamiltonian);

    // Cоздаем линдбладиан
    MKL_Complex16* lindbladian;
    GenerateLp(N, 2, lindbladian);

    // Cоздаем матрицу плотности
    MKL_Complex16* rho;
    GenerateDensity(N, 2, rho);

    // Вычисляем коэффициенты h
    std::vector<double> h_coeff = GetHCoef(hamiltonian, N);

    // // Вычисляем коэффициенты l
    // std::vector<MKL_Complex16> l_coeff = GetLCoef(lindbladian, N);

    // std::vector<MKL_Complex16> l_coeff_conjugate(l_coeff);
    // for (auto& elem : l_coeff_conjugate) {
    //     elem = Conjugate(elem);
    // }

    // Вычисляем матрицу Коссаковски
    // std::vector<MKL_Complex16> a(M * M);
    // for (int i = 0; i < M; ++i) {
    //     for (int j = 0; j < M; ++j) {
    //         a[i * M + j] = l_coeff[i] * l_coeff_conjugate[j];
    //     }
    // }

    auto f_tensor = GenerateTensorF(N);

    auto cmp = [](const std::pair<std::tuple<int, int, int>, double>& left,
                  const std::pair<std::tuple<int, int, int>, double>& right) {
        if (std::get<2>(left.first) == std::get<2>(right.first)) {
            return std::get<1>(left.first) < std::get<1>(right.first);
        }
        return std::get<2>(left.first) < std::get<2>(right.first);
    };

    std::sort(f_tensor.begin(), f_tensor.end(), cmp);

    auto& f_tensor_sorted = f_tensor;
    std::vector<double> values;
    values.reserve(f_tensor_sorted.size());
    std::vector<int> row_ind;
    row_ind.reserve(f_tensor_sorted.size());
    std::vector<int> col_ind;
    col_ind.reserve(f_tensor_sorted.size());

    values.push_back(h_coeff[std::get<0>(f_tensor_sorted[0].first)] * f_tensor_sorted[0].second);
    row_ind.push_back(std::get<2>(f_tensor_sorted[0].first));
    col_ind.push_back(std::get<1>(f_tensor_sorted[0].first));

    std::vector<std::pair<std::tuple<int, int>, double>> q_matrix;
    q_matrix.emplace_back(
        std::tuple(std::get<2>(f_tensor_sorted[0].first), std::get<1>(f_tensor_sorted[0].first)),
        h_coeff[std::get<0>(f_tensor_sorted[0].first)] * f_tensor_sorted[0].second);

    for (size_t i = 1; i < f_tensor_sorted.size(); ++i) {
        if (std::get<2>(f_tensor_sorted[i - 1].first) == std::get<2>(f_tensor_sorted[i].first) and
            std::get<1>(f_tensor_sorted[i - 1].first) == std::get<1>(f_tensor_sorted[i].first)) {
            values.back() +=
                h_coeff[std::get<0>(f_tensor_sorted[i].first)] * f_tensor_sorted[i].second;
            q_matrix.back().second +=
                h_coeff[std::get<0>(f_tensor_sorted[i].first)] * f_tensor_sorted[i].second;
        } else {
            values.push_back(h_coeff[std::get<0>(f_tensor_sorted[i].first)] *
                             f_tensor_sorted[i].second);
            row_ind.push_back(std::get<2>(f_tensor_sorted[i].first));
            col_ind.push_back(std::get<1>(f_tensor_sorted[i].first));
            q_matrix.emplace_back(
                std::tuple(std::get<2>(f_tensor_sorted[i].first),
                           std::get<1>(f_tensor_sorted[i].first)),
                h_coeff[std::get<0>(f_tensor_sorted[i].first)] * f_tensor_sorted[i].second);
        }
    }

    std::vector<std::pair<std::tuple<int, int>, double>> q_tensor;
    q_tensor.reserve(values.size());
    for (size_t i = 0; i < values.size(); ++i) {
        q_tensor.emplace_back(std::tuple(row_ind[i], col_ind[i]), values[i]);
        if (q_tensor[i] != q_matrix[i]) {
            std::cout << i << "\n";
        }
    }

    std::cout << q_tensor.size() << "\n";
}