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
void calculate_f(double t, const double* v, double* result,
                 const std::vector<std::pair<std::tuple<int, int>, double>>& q_matrix,
                 const double* r_matrix, const double* k_vector, double* workspace, int M) {
    double* A = workspace;

    // 1. Вычисляем матрицу A(t) = Q(t) + R
    cblas_dcopy(M * M, r_matrix, 1, workspace, 1);

    for (const auto& [tpl, nmb]: q_matrix) {
        size_t index = std::get<0>(tpl) * M + std::get<1>(tpl);
        A[index] += nmb;
    }

    // 2. Вычисляем A(t) * v
    // cblas_dgemv: result = alpha*A*v + beta*result
    // Мы хотим result = 1.0 * A * v + 0.0 * result
    cblas_dgemv(CblasRowMajor, CblasNoTrans, M, M,  // Размеры матрицы A
                1.0,                                // alpha
                A, M,                               // Матрица A и ее lda
                v, 1,                               // Вектор v и его инкремент
                0.0,                                // beta
                result, 1);                         // Результирующий вектор и его инкремент

    // 3. Добавляем вектор K
    // cblas_daxpy: result = 1.0 * K + result
    cblas_daxpy(M, 1.0, k_vector, 1, result, 1);
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

double* GetVCoef(MKL_Complex16* rho, int N) {
    double* v_coeff = (double*)mkl_malloc((N * N - 1) * sizeof(double), 64);
    size_t ind = 0;
    for (int j = 0; j < N; ++j) {
        for (int k = j + 1; k < N; ++k) {
            int index = j * N + k;
            v_coeff[ind] = sqrt(2.) * rho[index].real;
            ++ind;
        }
    }

    for (int j = 0; j < N; ++j) {
        for (int k = j + 1; k < N; ++k) {
            int index = k * N + j;
            v_coeff[ind] = sqrt(2.) * rho[index].imag;
            ++ind;
        }
    }

    for (int l = 0; l < N - 1; ++l) {
        double coeff = 0.;

        for (int k = 0; k < l + 1; ++k) {
            int index = k * N + k;
            coeff += rho[index].real / sqrt((l + 1) * (l + 2));
        }

        int index = (l + 1) * N + (l + 1);
        coeff += -sqrt(l + 1) * rho[index].real / sqrt(l + 2);

        v_coeff[ind] = coeff;
        ++ind;
    }

    return v_coeff;
}

// Проверка приближенной эрмитовости: M == M^† в пределах eps
double check_hermitian_approx(const MKL_Complex16* M, int d) {
    double result = 0.0;
    for (int i = 0; i < d; ++i) {
        for (int j = 0; j < d; ++j) {
            const MKL_Complex16& a = M[i * d + j];
            const MKL_Complex16& b = M[j * d + i];  // should be conj(a)
            double re_diff = a.real - b.real;
            double im_diff = a.imag + b.imag;  // a.imag - (-b.imag) since b should be conj(a)
            result = std::max(re_diff * re_diff + im_diff * im_diff, result);
        }
    }
    return result;
}

void FillCommutator(std::vector<std::vector<MKL_Complex16*>>* commut, int N) {
    auto& commutator = *commut;
    // 1 квадрант
    for (int j_k = 1; j_k + 1 < N; ++j_k) {
        for (int i = 0; i < j_k; ++i) {
            for (int l = j_k + 1; l < N; ++l) {
                size_t f_c = Mapping(i, j_k, N);
                size_t s_c = Mapping(j_k, l, N);
                commutator[f_c][s_c][i * N + l] = {0.5, 0};
                commutator[f_c][s_c][l * N + i] = {-0.5, 0};
            }
        }
    }

    for (int i_l = 1; i_l + 1 < N; ++i_l) {
        for (int j = i_l + 1; j < N; ++j) {
            for (int k = 0; k < i_l; ++k) {
                size_t f_c = Mapping(i_l, j, N);
                size_t s_c = Mapping(k, i_l, N);

                commutator[f_c][s_c][k * N + j] = {-0.5, 0};
                commutator[f_c][s_c][j * N + k] = {0.5, 0};
            }
        }
    }

    for (int i_k = 0; i_k + 1 < N; ++i_k) {
        for (int j = i_k + 1; j < N; ++j) {
            for (int l = i_k + 1; l < N; ++l) {
                if (j != l) {
                    size_t f_c = Mapping(i_k, j, N);
                    size_t s_c = Mapping(i_k, l, N);

                    commutator[f_c][s_c][j * N + l] = {0.5, 0};
                    commutator[f_c][s_c][l * N + j] = {-0.5, 0};
                }
            }
        }
    }

    for (int j_l = 1; j_l < N; ++j_l) {
        for (int i = 0; i < j_l; ++i) {
            for (int k = 0; k < j_l; ++k) {
                if (i != k) {
                    size_t f_c = Mapping(i, j_l, N);
                    size_t s_c = Mapping(k, j_l, N);

                    commutator[f_c][s_c][i * N + k] = {0.5, 0};
                    commutator[f_c][s_c][k * N + i] = {-0.5, 0};
                }
            }
        }
    }

    // 2 квадрант и 4 квадрант

    // i < l
    for (int j_k = 1; j_k + 1 < N; ++j_k) {
        for (int i = 0; i < j_k; ++i) {
            for (int l = j_k + 1; l < N; ++l) {
                size_t f_c = Mapping(i, j_k, N);
                size_t s_c = N * (N - 1) / 2 + Mapping(j_k, l, N);
                commutator[f_c][s_c][i * N + l] = {0, -0.5};
                commutator[f_c][s_c][l * N + i] = {0, -0.5};

                commutator[s_c][f_c][i * N + l] = {0, 0.5};
                commutator[s_c][f_c][l * N + i] = {0, 0.5};
            }
        }
    }

    // k < j
    for (int i_l = 1; i_l + 1 < N; ++i_l) {
        for (int j = i_l + 1; j < N; ++j) {
            for (int k = 0; k < i_l; ++k) {
                size_t f_c = Mapping(i_l, j, N);
                size_t s_c = N * (N - 1) / 2 + Mapping(k, i_l, N);

                commutator[f_c][s_c][k * N + j] = {0, 0.5};
                commutator[f_c][s_c][j * N + k] = {0, 0.5};

                commutator[s_c][f_c][k * N + j] = {0, -0.5};
                commutator[s_c][f_c][j * N + k] = {0, -0.5};
            }
        }
    }

    for (int i_k = 0; i_k + 1 < N; ++i_k) {
        for (int j = i_k + 1; j < N; ++j) {
            for (int l = i_k + 1; l < N; ++l) {
                if (j != l) {
                    size_t f_c = Mapping(i_k, j, N);
                    size_t s_c = N * (N - 1) / 2 + Mapping(i_k, l, N);

                    commutator[f_c][s_c][j * N + l] = {0, -0.5};
                    commutator[f_c][s_c][l * N + j] = {0, -0.5};

                    commutator[s_c][f_c][j * N + l] = {0, 0.5};
                    commutator[s_c][f_c][l * N + j] = {0, 0.5};
                }
            }
        }
    }

    for (int j_l = 1; j_l < N; ++j_l) {
        for (int i = 0; i < j_l; ++i) {
            for (int k = 0; k < j_l; ++k) {
                if (i != k) {
                    size_t f_c = Mapping(i, j_l, N);
                    size_t s_c = N * (N - 1) / 2 + Mapping(k, j_l, N);

                    commutator[f_c][s_c][i * N + k] = {0, 0.5};
                    commutator[f_c][s_c][k * N + i] = {0, 0.5};

                    commutator[s_c][f_c][i * N + k] = {0, -0.5};
                    commutator[s_c][f_c][k * N + i] = {0, -0.5};
                }
            }
        }
    }

    for (int i_k = 0; i_k + 1 < N; ++i_k) {
        for (int j_l = i_k + 1; j_l < N; ++j_l) {
            size_t f_c = Mapping(i_k, j_l, N);
            size_t s_c = N * (N - 1) / 2 + Mapping(i_k, j_l, N);

            commutator[f_c][s_c][i_k * N + i_k] = {0, 1.};
            commutator[f_c][s_c][j_l * N + j_l] = {0, -1.};

            commutator[s_c][f_c][i_k * N + i_k] = {0, -1.};
            commutator[s_c][f_c][j_l * N + j_l] = {0, 1.};
        }
    }

    // 3 квадрант и 7 квадрант
    for (int l = 0; l < N - 1; ++l) {

        // j = l + 1
        for (int i = 0; i < l + 1; ++i) {
            size_t f_c = Mapping(i, l + 1, N);
            size_t s_c = N * (N - 1) + l;
            commutator[f_c][s_c][i * N + l + 1] = {-sqrt(0.5 * (l + 2) / (l + 1)), 0};
            commutator[f_c][s_c][(l + 1) * N + i] = {sqrt(0.5 * (l + 2) / (l + 1)), 0};

            commutator[s_c][f_c][i * N + l + 1] = {sqrt(0.5 * (l + 2) / (l + 1)), 0};
            commutator[s_c][f_c][(l + 1) * N + i] = {-sqrt(0.5 * (l + 2) / (l + 1)), 0};
        }

        for (int i = 0; i < l + 1; ++i) {
            for (int j = l + 2; j < N; ++j) {
                size_t f_c = Mapping(i, j, N);
                size_t s_c = N * (N - 1) + l;
                commutator[f_c][s_c][i * N + j] = {-1. / sqrt(2. * (l + 1) * (l + 2)), 0};
                commutator[f_c][s_c][j * N + i] = {1. / sqrt(2. * (l + 1) * (l + 2)), 0};

                commutator[s_c][f_c][i * N + j] = {1. / sqrt(2. * (l + 1) * (l + 2)), 0};
                commutator[s_c][f_c][j * N + i] = {-1. / sqrt(2. * (l + 1) * (l + 2)), 0};
            }
        }

        // i = l + 1
        for (int j = l + 2; j < N; ++j) {
            size_t f_c = Mapping(l + 1, j, N);
            size_t s_c = N * (N - 1) + l;
            commutator[f_c][s_c][(l + 1) * N + j] = {sqrt(0.5 * (l + 1) / (l + 2)), 0};
            commutator[f_c][s_c][j * N + (l + 1)] = {-sqrt(0.5 * (l + 1) / (l + 2)), 0};

            commutator[s_c][f_c][(l + 1) * N + j] = {-sqrt(0.5 * (l + 1) / (l + 2)), 0};
            commutator[s_c][f_c][j * N + (l + 1)] = {sqrt(0.5 * (l + 1) / (l + 2)), 0};
        }
    }

    // 5 квадрант
    for (int j_k = 1; j_k + 1 < N; ++j_k) {
        for (int i = 0; i < j_k; ++i) {
            for (int l = j_k + 1; l < N; ++l) {
                size_t f_c = N * (N - 1) / 2 + Mapping(i, j_k, N);
                size_t s_c = N * (N - 1) / 2 + Mapping(j_k, l, N);
                commutator[f_c][s_c][i * N + l] = {-0.5, 0};
                commutator[f_c][s_c][l * N + i] = {0.5, 0};
            }
        }
    }

    for (int i_l = 1; i_l + 1 < N; ++i_l) {
        for (int j = i_l + 1; j < N; ++j) {
            for (int k = 0; k < i_l; ++k) {
                size_t f_c = N * (N - 1) / 2 + Mapping(i_l, j, N);
                size_t s_c = N * (N - 1) / 2 + Mapping(k, i_l, N);

                commutator[f_c][s_c][k * N + j] = {0.5, 0};
                commutator[f_c][s_c][j * N + k] = {-0.5, 0};
            }
        }
    }

    for (int i_k = 0; i_k + 1 < N; ++i_k) {
        for (int j = i_k + 1; j < N; ++j) {
            for (int l = i_k + 1; l < N; ++l) {
                if (j != l) {
                    size_t f_c = N * (N - 1) / 2 + Mapping(i_k, j, N);
                    size_t s_c = N * (N - 1) / 2 + Mapping(i_k, l, N);

                    commutator[f_c][s_c][j * N + l] = {0.5, 0};
                    commutator[f_c][s_c][l * N + j] = {-0.5, 0};
                }
            }
        }
    }

    for (int j_l = 1; j_l < N; ++j_l) {
        for (int i = 0; i < j_l; ++i) {
            for (int k = 0; k < j_l; ++k) {
                if (i != k) {
                    size_t f_c = N * (N - 1) / 2 + Mapping(i, j_l, N);
                    size_t s_c = N * (N - 1) / 2 + Mapping(k, j_l, N);

                    commutator[f_c][s_c][i * N + k] = {0.5, 0};
                    commutator[f_c][s_c][k * N + i] = {-0.5, 0};
                }
            }
        }
    }

    // 6, 8 квадрант

    for (int l = 0; l < N - 1; ++l) {

        // j = l + 1
        for (int i = 0; i < l + 1; ++i) {
            size_t f_c = N * (N - 1) / 2 + Mapping(i, l + 1, N);
            size_t s_c = N * (N - 1) + l;
            commutator[f_c][s_c][i * N + l + 1] = {0, sqrt(0.5 * (l + 2) / (l + 1))};
            commutator[f_c][s_c][(l + 1) * N + i] = {0, sqrt(0.5 * (l + 2) / (l + 1))};

            commutator[s_c][f_c][i * N + l + 1] = {0, -sqrt(0.5 * (l + 2) / (l + 1))};
            commutator[s_c][f_c][(l + 1) * N + i] = {0, -sqrt(0.5 * (l + 2) / (l + 1))};
        }

        for (int i = 0; i < l + 1; ++i) {
            for (int j = l + 2; j < N; ++j) {
                size_t f_c = N * (N - 1) / 2 + Mapping(i, j, N);
                size_t s_c = N * (N - 1) + l;
                commutator[f_c][s_c][i * N + j] = {0, 1. / sqrt(2. * (l + 1) * (l + 2))};
                commutator[f_c][s_c][j * N + i] = {0, 1. / sqrt(2. * (l + 1) * (l + 2))};

                commutator[s_c][f_c][i * N + j] = {0, -1. / sqrt(2. * (l + 1) * (l + 2))};
                commutator[s_c][f_c][j * N + i] = {0, -1. / sqrt(2. * (l + 1) * (l + 2))};
            }
        }

        // i = l + 1
        for (int j = l + 2; j < N; ++j) {
            size_t f_c = N * (N - 1) / 2 + Mapping(l + 1, j, N);
            size_t s_c = N * (N - 1) + l;
            commutator[f_c][s_c][(l + 1) * N + j] = {0, -sqrt(0.5 * (l + 1) / (l + 2))};
            commutator[f_c][s_c][j * N + (l + 1)] = {0, -sqrt(0.5 * (l + 1) / (l + 2))};

            commutator[s_c][f_c][(l + 1) * N + j] = {0, sqrt(0.5 * (l + 1) / (l + 2))};
            commutator[s_c][f_c][j * N + (l + 1)] = {0, sqrt(0.5 * (l + 1) / (l + 2))};
        }
    }
}

void FillAntiCommutator(std::vector<std::vector<MKL_Complex16*>>* anticommut, int N) {
    auto& anticommutator = *anticommut;
    // 1 квадрант
    for (int j_k = 1; j_k + 1 < N; ++j_k) {
        for (int i = 0; i < j_k; ++i) {
            for (int l = j_k + 1; l < N; ++l) {
                size_t f_c = Mapping(i, j_k, N);
                size_t s_c = Mapping(j_k, l, N);
                anticommutator[f_c][s_c][i * N + l] = {0.5, 0};
                anticommutator[f_c][s_c][l * N + i] = {0.5, 0};
            }
        }
    }

    for (int i_l = 1; i_l + 1 < N; ++i_l) {
        for (int j = i_l + 1; j < N; ++j) {
            for (int k = 0; k < i_l; ++k) {
                size_t f_c = Mapping(i_l, j, N);
                size_t s_c = Mapping(k, i_l, N);

                anticommutator[f_c][s_c][k * N + j] = {0.5, 0};
                anticommutator[f_c][s_c][j * N + k] = {0.5, 0};
            }
        }
    }

    for (int i_k = 0; i_k + 1 < N; ++i_k) {
        for (int j = i_k + 1; j < N; ++j) {
            for (int l = i_k + 1; l < N; ++l) {
                if (j != l) {
                    size_t f_c = Mapping(i_k, j, N);
                    size_t s_c = Mapping(i_k, l, N);

                    anticommutator[f_c][s_c][j * N + l] = {0.5, 0};
                    anticommutator[f_c][s_c][l * N + j] = {0.5, 0};
                }
            }
        }
    }

    for (int j_l = 1; j_l < N; ++j_l) {
        for (int i = 0; i < j_l; ++i) {
            for (int k = 0; k < j_l; ++k) {
                if (i != k) {
                    size_t f_c = Mapping(i, j_l, N);
                    size_t s_c = Mapping(k, j_l, N);

                    anticommutator[f_c][s_c][i * N + k] = {0.5, 0};
                    anticommutator[f_c][s_c][k * N + i] = {0.5, 0};
                }
            }
        }
    }

    for (int i_k = 0; i_k + 1 < N; ++i_k) {
        for (int j_l = i_k + 1; j_l < N; ++j_l) {
            size_t f_c = Mapping(i_k, j_l, N);
            size_t s_c = Mapping(i_k, j_l, N);

            anticommutator[f_c][s_c][i_k * N + i_k] = {1, 0};
            anticommutator[f_c][s_c][j_l * N + j_l] = {1, 0};
        }
    }

    // 2 квадрант и 4 квадрант

    // i < l
    for (int j_k = 1; j_k + 1 < N; ++j_k) {
        for (int i = 0; i < j_k; ++i) {
            for (int l = j_k + 1; l < N; ++l) {
                size_t f_c = Mapping(i, j_k, N);
                size_t s_c = N * (N - 1) / 2 + Mapping(j_k, l, N);
                anticommutator[f_c][s_c][i * N + l] = {0, -0.5};
                anticommutator[f_c][s_c][l * N + i] = {0, 0.5};

                anticommutator[s_c][f_c][i * N + l] = {0, -0.5};
                anticommutator[s_c][f_c][l * N + i] = {0, 0.5};
            }
        }
    }

    // k < j
    for (int i_l = 1; i_l + 1 < N; ++i_l) {
        for (int j = i_l + 1; j < N; ++j) {
            for (int k = 0; k < i_l; ++k) {
                size_t f_c = Mapping(i_l, j, N);
                size_t s_c = N * (N - 1) / 2 + Mapping(k, i_l, N);

                anticommutator[f_c][s_c][k * N + j] = {0, -0.5};
                anticommutator[f_c][s_c][j * N + k] = {0, 0.5};

                anticommutator[s_c][f_c][k * N + j] = {0, -0.5};
                anticommutator[s_c][f_c][j * N + k] = {0, 0.5};
            }
        }
    }

    for (int i_k = 0; i_k + 1 < N; ++i_k) {
        for (int j = i_k + 1; j < N; ++j) {
            for (int l = i_k + 1; l < N; ++l) {
                if (j != l) {
                    size_t f_c = Mapping(i_k, j, N);
                    size_t s_c = N * (N - 1) / 2 + Mapping(i_k, l, N);

                    anticommutator[f_c][s_c][j * N + l] = {0, -0.5};
                    anticommutator[f_c][s_c][l * N + j] = {0, 0.5};

                    anticommutator[s_c][f_c][j * N + l] = {0, -0.5};
                    anticommutator[s_c][f_c][l * N + j] = {0, 0.5};
                }
            }
        }
    }

    for (int j_l = 1; j_l < N; ++j_l) {
        for (int i = 0; i < j_l; ++i) {
            for (int k = 0; k < j_l; ++k) {
                if (i != k) {
                    size_t f_c = Mapping(i, j_l, N);
                    size_t s_c = N * (N - 1) / 2 + Mapping(k, j_l, N);

                    anticommutator[f_c][s_c][i * N + k] = {0, 0.5};
                    anticommutator[f_c][s_c][k * N + i] = {0, -0.5};

                    anticommutator[s_c][f_c][i * N + k] = {0, 0.5};
                    anticommutator[s_c][f_c][k * N + i] = {0, -0.5};
                }
            }
        }
    }

    // 3 квадрант и 7 квадрант
    for (int l = 0; l < N - 1; ++l) {

        // j < l + 1
        for (int j = 1; j < l + 1; ++j) {
            for (int i = 0; i < j; ++i) {
                size_t f_c = Mapping(i, j, N);
                size_t s_c = N * (N - 1) + l;
                anticommutator[f_c][s_c][i * N + j] = {sqrt(2) / sqrt((l + 1) * (l + 2)), 0};
                anticommutator[f_c][s_c][j * N + i] = {sqrt(2) / sqrt((l + 1) * (l + 2)), 0};

                anticommutator[s_c][f_c][i * N + j] = {sqrt(2) / sqrt((l + 1) * (l + 2)), 0};
                anticommutator[s_c][f_c][j * N + i] = {sqrt(2) / sqrt((l + 1) * (l + 2)), 0};
            }
        }

        // j = l + 1
        for (int i = 0; i < l + 1; ++i) {
            size_t f_c = Mapping(i, l + 1, N);
            size_t s_c = N * (N - 1) + l;
            anticommutator[f_c][s_c][i * N + l + 1] = {-l / sqrt(2 * (l + 1) * (l + 2)), 0};
            anticommutator[f_c][s_c][(l + 1) * N + i] = {-l / sqrt(2 * (l + 1) * (l + 2)), 0};

            anticommutator[s_c][f_c][i * N + l + 1] = {-l / sqrt(2 * (l + 1) * (l + 2)), 0};
            anticommutator[s_c][f_c][(l + 1) * N + i] = {-l / sqrt(2 * (l + 1) * (l + 2)), 0};
        }

        // i < l + 1 < j
        for (int i = 0; i < l + 1; ++i) {
            for (int j = l + 2; j < N; ++j) {
                size_t f_c = Mapping(i, j, N);
                size_t s_c = N * (N - 1) + l;
                anticommutator[f_c][s_c][i * N + j] = {1. / sqrt(2. * (l + 1) * (l + 2)), 0};
                anticommutator[f_c][s_c][j * N + i] = {1. / sqrt(2. * (l + 1) * (l + 2)), 0};

                anticommutator[s_c][f_c][i * N + j] = {1. / sqrt(2. * (l + 1) * (l + 2)), 0};
                anticommutator[s_c][f_c][j * N + i] = {1. / sqrt(2. * (l + 1) * (l + 2)), 0};
            }
        }

        // i = l + 1
        for (int j = l + 2; j < N; ++j) {
            size_t f_c = Mapping(l + 1, j, N);
            size_t s_c = N * (N - 1) + l;
            anticommutator[f_c][s_c][(l + 1) * N + j] = {-sqrt(0.5 * (l + 1) / (l + 2)), 0};
            anticommutator[f_c][s_c][j * N + (l + 1)] = {-sqrt(0.5 * (l + 1) / (l + 2)), 0};

            anticommutator[s_c][f_c][(l + 1) * N + j] = {-sqrt(0.5 * (l + 1) / (l + 2)), 0};
            anticommutator[s_c][f_c][j * N + (l + 1)] = {-sqrt(0.5 * (l + 1) / (l + 2)), 0};
        }
    }

    // 5 квадрант
    for (int j_k = 1; j_k + 1 < N; ++j_k) {
        for (int i = 0; i < j_k; ++i) {
            for (int l = j_k + 1; l < N; ++l) {
                size_t f_c = N * (N - 1) / 2 + Mapping(i, j_k, N);
                size_t s_c = N * (N - 1) / 2 + Mapping(j_k, l, N);
                anticommutator[f_c][s_c][i * N + l] = {-0.5, 0};
                anticommutator[f_c][s_c][l * N + i] = {-0.5, 0};
            }
        }
    }

    for (int i_l = 1; i_l + 1 < N; ++i_l) {
        for (int j = i_l + 1; j < N; ++j) {
            for (int k = 0; k < i_l; ++k) {
                size_t f_c = N * (N - 1) / 2 + Mapping(i_l, j, N);
                size_t s_c = N * (N - 1) / 2 + Mapping(k, i_l, N);

                anticommutator[f_c][s_c][k * N + j] = {-0.5, 0};
                anticommutator[f_c][s_c][j * N + k] = {-0.5, 0};
            }
        }
    }

    for (int i_k = 0; i_k + 1 < N; ++i_k) {
        for (int j = i_k + 1; j < N; ++j) {
            for (int l = i_k + 1; l < N; ++l) {
                if (j != l) {
                    size_t f_c = N * (N - 1) / 2 + Mapping(i_k, j, N);
                    size_t s_c = N * (N - 1) / 2 + Mapping(i_k, l, N);

                    anticommutator[f_c][s_c][j * N + l] = {0.5, 0};
                    anticommutator[f_c][s_c][l * N + j] = {0.5, 0};
                }
            }
        }
    }

    for (int j_l = 1; j_l < N; ++j_l) {
        for (int i = 0; i < j_l; ++i) {
            for (int k = 0; k < j_l; ++k) {
                if (i != k) {
                    size_t f_c = N * (N - 1) / 2 + Mapping(i, j_l, N);
                    size_t s_c = N * (N - 1) / 2 + Mapping(k, j_l, N);

                    anticommutator[f_c][s_c][i * N + k] = {0.5, 0};
                    anticommutator[f_c][s_c][k * N + i] = {0.5, 0};
                }
            }
        }
    }

    for (int i_k = 0; i_k + 1 < N; ++i_k) {
        for (int j_l = i_k + 1; j_l < N; ++j_l) {
            size_t f_c = N * (N - 1) / 2 + Mapping(i_k, j_l, N);
            size_t s_c = N * (N - 1) / 2 + Mapping(i_k, j_l, N);

            anticommutator[f_c][s_c][i_k * N + i_k] = {1, 0};
            anticommutator[f_c][s_c][j_l * N + j_l] = {1, 0};
        }
    }

    // 6, 8 квадрант

    for (int l = 0; l < N - 1; ++l) {

        // j < l + 1
        for (int j = 1; j < l + 1; ++j) {
            for (int i = 0; i < j; ++i) {
                size_t f_c = N * (N - 1) / 2 + Mapping(i, j, N);
                size_t s_c = N * (N - 1) + l;
                anticommutator[f_c][s_c][i * N + j] = {0, -sqrt(2) / sqrt((l + 1) * (l + 2))};
                anticommutator[f_c][s_c][j * N + i] = {0, sqrt(2) / sqrt((l + 1) * (l + 2))};

                anticommutator[s_c][f_c][i * N + j] = {0, -sqrt(2) / sqrt((l + 1) * (l + 2))};
                anticommutator[s_c][f_c][j * N + i] = {0, sqrt(2) / sqrt((l + 1) * (l + 2))};
            }
        }

        // j = l + 1
        for (int i = 0; i < l + 1; ++i) {
            size_t f_c = N * (N - 1) / 2 + Mapping(i, l + 1, N);
            size_t s_c = N * (N - 1) + l;
            anticommutator[f_c][s_c][i * N + l + 1] = {0, l / sqrt(2 * (l + 1) * (l + 2))};
            anticommutator[f_c][s_c][(l + 1) * N + i] = {0, -l / sqrt(2 * (l + 1) * (l + 2))};

            anticommutator[s_c][f_c][i * N + l + 1] = {0, l / sqrt(2 * (l + 1) * (l + 2))};
            anticommutator[s_c][f_c][(l + 1) * N + i] = {0, -l / sqrt(2 * (l + 1) * (l + 2))};
        }

        // i < l + 1 < j
        for (int i = 0; i < l + 1; ++i) {
            for (int j = l + 2; j < N; ++j) {
                size_t f_c = N * (N - 1) / 2 + Mapping(i, j, N);
                size_t s_c = N * (N - 1) + l;

                anticommutator[f_c][s_c][i * N + j] = {0, -1. / sqrt(2. * (l + 1) * (l + 2))};
                anticommutator[f_c][s_c][j * N + i] = {0, 1. / sqrt(2. * (l + 1) * (l + 2))};

                anticommutator[s_c][f_c][i * N + j] = {0, -1. / sqrt(2. * (l + 1) * (l + 2))};
                anticommutator[s_c][f_c][j * N + i] = {0, 1. / sqrt(2. * (l + 1) * (l + 2))};
            }
        }

        // i = l + 1
        for (int j = l + 2; j < N; ++j) {
            size_t f_c = N * (N - 1) / 2 + Mapping(l + 1, j, N);
            size_t s_c = N * (N - 1) + l;
            anticommutator[f_c][s_c][(l + 1) * N + j] = {0, sqrt(0.5 * (l + 1) / (l + 2))};
            anticommutator[f_c][s_c][j * N + (l + 1)] = {0, -sqrt(0.5 * (l + 1) / (l + 2))};

            anticommutator[s_c][f_c][(l + 1) * N + j] = {0, sqrt(0.5 * (l + 1) / (l + 2))};
            anticommutator[s_c][f_c][j * N + (l + 1)] = {0, -sqrt(0.5 * (l + 1) / (l + 2))};
        }
    }

    // 9 квадрант
    for (int l = 0; l < N - 1; ++l) {
        for (int m = 0; m < N - 1; ++m) {
            size_t f_c = N * (N - 1) + l;
            size_t s_c = N * (N - 1) + m;

            if (l < m) {
                for (int k = 0; k < l + 1; ++k) {
                    int index = k * N + k;
                    anticommutator[f_c][s_c][index] = {
                        2. / (sqrt((m + 1) * (m + 2)) * sqrt((l + 1) * (l + 2))), 0.};
                }
                int index = (l + 1) * N + (l + 1);
                anticommutator[f_c][s_c][index] = {
                    -2. * sqrt(l + 1) / (sqrt((m + 1) * (m + 2)) * sqrt(l + 2)), 0.};

            } else if (l == m) {
                for (int k = 0; k < l + 1; ++k) {
                    int index = k * N + k;
                    anticommutator[f_c][s_c][index] = {2. / ((l + 1) * (l + 2)), 0.};
                }
                int index = (l + 1) * N + (l + 1);
                anticommutator[f_c][s_c][index] = {2. * (l + 1) / (l + 2), 0.};

            } else {
                for (int k = 0; k < m + 1; ++k) {
                    int index = k * N + k;
                    anticommutator[f_c][s_c][index] = {
                        2. / (sqrt((m + 1) * (m + 2)) * sqrt((l + 1) * (l + 2))), 0.};
                }
                int index = (m + 1) * N + (m + 1);
                anticommutator[f_c][s_c][index] = {
                    -2. * sqrt(m + 1) / (sqrt((l + 1) * (l + 2)) * sqrt(m + 2)), 0.};
            }
        }
    }
}

int main() {
    // Параметры
    int N = 5;
    int M = N * N - 1;

    // создаем гамильтониан и заполняем его нулями
    MKL_Complex16* hamiltonian;
    GenerateTracelessHamiltonian(N, 2, hamiltonian);

    // создаем линдбладиан и заполняем его нулями
    MKL_Complex16* lindbladian;
    GenerateLp(N, 2, lindbladian);

    MKL_Complex16* rho;
    GenerateDensity(N, 2, rho);

    // вычисляем коэффициенты h
    std::vector<double> h_coeff = GetHCoef(hamiltonian, N);
    // вычисляем коэффициенты l
    std::vector<MKL_Complex16> l_coeff = GetLCoef(lindbladian, N);

    std::vector<MKL_Complex16> l_coeff_conjugate(l_coeff);
    for (auto& elem : l_coeff_conjugate) {
        elem = Conjugate(elem);
    }

    // Вычисляем матрицу Коссаковски
    std::vector<MKL_Complex16> a(M * M);
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < M; ++j) {
            a[i * M + j] = l_coeff[i] * l_coeff_conjugate[j];
        }
    }

    auto f_tensor = GenerateTensorF(N);
    auto d_tensor = GenerateTensorD(N);
    auto z_tensor = GenerateTensorZ(f_tensor, d_tensor);

    // сортировка по (s, n) тензора (m, n, s)
    auto cmp = [](const std::pair<std::tuple<int, int, int>, double>& left,
                  const std::pair<std::tuple<int, int, int>, double>& right) {
        if (std::get<2>(left.first) == std::get<2>(right.first)) {
            return std::get<1>(left.first) < std::get<1>(right.first);
        }
        return std::get<2>(left.first) < std::get<2>(right.first);
    };
    std::sort(f_tensor.begin(), f_tensor.end(), cmp);

    auto q_matrix = GenerateCOOMatrixQ(&f_tensor, h_coeff);

    double* k_vector = GenerateVectorK(N, a, f_tensor);

    double* r_matrix = GenerateMatrixR(N, l_coeff, l_coeff_conjugate, &f_tensor, &z_tensor);

    print_double_matrix_rowmajor(r_matrix, M, "r_matrix");

    // Начальные условия
    double t0 = 0.0;

    // Временные векторы для РК4
    double* k1 = (double*)mkl_malloc(M * sizeof(double), 64);
    double* k2 = (double*)mkl_malloc(M * sizeof(double), 64);
    double* k3 = (double*)mkl_malloc(M * sizeof(double), 64);
    double* k4 = (double*)mkl_malloc(M * sizeof(double), 64);
    double* v_temp = (double*)mkl_malloc(M * sizeof(double), 64);
    double* v_sum = (double*)mkl_malloc(M * sizeof(double), 64);

    // Параметры
    double t_end = 1.3;
    double h = 0.01;

    double t = t0;
    std::cout << std::fixed << std::setprecision(6);
    double* workspace = (double*)mkl_malloc(M * M * sizeof(double), 64);
    double* v = GetVCoef(rho, N);
    while (t < t_end + h / 2) {

        // --- Шаг метода Рунге-Кутты 4-го порядка ---

        // k1 = f(t, v)
        calculate_f(t, v, k1, q_matrix, r_matrix, k_vector, workspace, M);

        // k2 = f(t + h/2, v + h/2 * k1)
        cblas_dcopy(M, v, 1, v_temp, 1);            // v_temp = v
        cblas_daxpy(M, 0.5 * h, k1, 1, v_temp, 1);  // v_temp = v + 0.5*h*k1
        calculate_f(t + 0.5 * h, v_temp, k2, q_matrix, r_matrix, k_vector, workspace, M);

        // k3 = f(t + h/2, v + h/2 * k2)
        cblas_dcopy(M, v, 1, v_temp, 1);            // v_temp = v
        cblas_daxpy(M, 0.5 * h, k2, 1, v_temp, 1);  // v_temp = v + 0.5*h*k2
        calculate_f(t + 0.5 * h, v_temp, k3, q_matrix, r_matrix, k_vector, workspace, M);

        // k4 = f(t + h, v + h * k3)
        cblas_dcopy(M, v, 1, v_temp, 1);      // v_temp = v
        cblas_daxpy(M, h, k3, 1, v_temp, 1);  // v_temp = v + h*k3
        calculate_f(t + h, v_temp, k4, q_matrix, r_matrix, k_vector, workspace, M);

        // Обновляем v: v = v + (h/6) * (k1 + 2k2 + 2k3 + k4)
        cblas_dcopy(M, k1, 1, v_sum, 1);          // v_sum = k1
        cblas_daxpy(M, 2.0, k2, 1, v_sum, 1);     // v_sum = k1 + 2*k2
        cblas_daxpy(M, 2.0, k3, 1, v_sum, 1);     // v_sum = k1 + 2*k2 + 2*k3
        cblas_daxpy(M, 1.0, k4, 1, v_sum, 1);     // v_sum = k1 + 2*k2 + 2*k3 + k4
        cblas_daxpy(M, h / 6.0, v_sum, 1, v, 1);  // v = v + (h/6)*v_sum

        print_vector("v", t, v, M);

        t += h;
    }

    // // Вычисляем разницу в секундах (как double)
    // std::cout << "Интегрирование ДУ: "
    //           << duration_cast<duration<double>>(high_resolution_clock::now() - start).count()
    //           << "\n";

    // MKL_Complex16* matrix_rho = (MKL_Complex16*)mkl_malloc(N * N * sizeof(MKL_Complex16), 64);
    // memset(matrix_rho, 0, N * N * sizeof(MKL_Complex16));

    // for (int i = 0; i < N * N; ++i) {
    //     for (int j = 0; j < M; ++j) {
    //         matrix_rho[i] += v[j] * basis_array[j][i];
    //     }
    // }

    // // print_matrix_rowmajor(matrix_rho, N, "result");
    // MKL_Complex16 tr = Trace(matrix_rho, N);

    // std::cout << std::fixed << std::setprecision(17) << tr.real << "\n";

    // std::cout << std::fixed << std::setprecision(17) << check_hermitian_approx(matrix_rho, N)
    //           << "\n";

    // // --- Очистка памяти ---
    // // Вектор сам удалит массив указателей, но нам нужно вручную освободить
    // // данные, на которые они указывают!
    // for (MKL_Complex16* mat_ptr : basis_array) {
    //     mkl_free(mat_ptr);
    // }
    // // mkl_free(middle_result);
    // mkl_free(end_result);
    // mkl_free(hamiltonian);
    // mkl_free(lindbladian);
    // mkl_free(rho);
    // mkl_free(matrix_rho);
    // mkl_free(f_tensor);
    // mkl_free(d_tensor);
    // mkl_free(z_tensor);
    // mkl_free(q_tensor);
    // mkl_free(k_tensor);
    // mkl_free(r_tensor);

    // // Освобождение памяти
    // mkl_free(v);
    // mkl_free(k1);
    // mkl_free(k2);
    // mkl_free(k3);
    // mkl_free(k4);
    // mkl_free(v_temp);
    // mkl_free(v_sum);

    return 0;
}
