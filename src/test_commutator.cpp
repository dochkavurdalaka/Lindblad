
#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <cstring>
#include <chrono>
#include "mkl.h"
#include "mkl_complex16.h"
#include "generate_matrices.h"

using namespace std::chrono;

MKL_Complex16 Trace(const MKL_Complex16 *matrix, int n) {
    MKL_Complex16 tr = {0.0, 0.0};
    for (int i = 0; i < n; ++i) {
        int diag_idx = i * n + i;
        tr.real += matrix[diag_idx].real;
        tr.imag += matrix[diag_idx].imag;
    }
    return tr;
}

MKL_Complex16 Conjugate(MKL_Complex16 number) {
    number.imag = -number.imag;
    return number;
}

void Commutator(const MKL_Complex16 *F_m, const MKL_Complex16 *F_n, MKL_Complex16 *dummy_result,
                int N) {
    MKL_Complex16 alpha = {1.0, 0.0}, beta = {0.0, 0.0};
    cblas_zgemm(CblasRowMajor,  // Указывает, что матрицы хранятся построчно
                                // (стандарт для C/C++)
                CblasNoTrans,   // Операция для A: не транспонировать
                CblasNoTrans,   // Операция для B: не транспонировать
                N,              // Количество строк в матрице A (и C)
                N,              // Количество столбцов в матрице B (и C)
                N,              // Количество столбцов в A и строк в B
                &alpha,         // Указатель на скаляр alpha
                F_m,            // Матрица A
                N,              // Ведущий размер (leading dimension) для A. Для RowMajor это
                                // количество столбцов.
                F_n,            // Матрица B
                N,              // Ведущий размер для B.
                &beta,          // Указатель на скаляр beta
                dummy_result,   // Матрица C (результат)
                N               // Ведущий размер для C.
    );
    alpha = {-1.0, 0.0};
    beta = {1.0, 0.0};
    cblas_zgemm(CblasRowMajor,  // Указывает, что матрицы хранятся построчно
                                // (стандарт для C/C++)
                CblasNoTrans,   // Операция для A: не транспонировать
                CblasNoTrans,   // Операция для B: не транспонировать
                N,              // Количество строк в матрице A (и C)
                N,              // Количество столбцов в матрице B (и C)
                N,              // Количество столбцов в A и строк в B
                &alpha,         // Указатель на скаляр alpha
                F_n,            // Матрица A
                N,              // Ведущий размер (leading dimension) для A. Для RowMajor это
                                // количество столбцов.
                F_m,            // Матрица B
                N,              // Ведущий размер для B.
                &beta,          // Указатель на скаляр beta
                dummy_result,   // Матрица C (результат)
                N               // Ведущий размер для C.
    );
}

void AntiCommutator(const MKL_Complex16 *F_m, const MKL_Complex16 *F_n, MKL_Complex16 *dummy_result,
                    int N) {
    MKL_Complex16 alpha = {1.0, 0.0}, beta = {0.0, 0.0};
    cblas_zgemm(CblasRowMajor,  // Указывает, что матрицы хранятся построчно
                                // (стандарт для C/C++)
                CblasNoTrans,   // Операция для A: не транспонировать
                CblasNoTrans,   // Операция для B: не транспонировать
                N,              // Количество строк в матрице A (и C)
                N,              // Количество столбцов в матрице B (и C)
                N,              // Количество столбцов в A и строк в B
                &alpha,         // Указатель на скаляр alpha
                F_m,            // Матрица A
                N,              // Ведущий размер (leading dimension) для A. Для RowMajor это
                                // количество столбцов.
                F_n,            // Матрица B
                N,              // Ведущий размер для B.
                &beta,          // Указатель на скаляр beta
                dummy_result,   // Матрица C (результат)
                N               // Ведущий размер для C.
    );
    beta = {1.0, 0.0};
    cblas_zgemm(CblasRowMajor,  // Указывает, что матрицы хранятся построчно
                                // (стандарт для C/C++)
                CblasNoTrans,   // Операция для A: не транспонировать
                CblasNoTrans,   // Операция для B: не транспонировать
                N,              // Количество строк в матрице A (и C)
                N,              // Количество столбцов в матрице B (и C)
                N,              // Количество столбцов в A и строк в B
                &alpha,         // Указатель на скаляр alpha
                F_n,            // Матрица A
                N,              // Ведущий размер (leading dimension) для A. Для RowMajor это
                                // количество столбцов.
                F_m,            // Матрица B
                N,              // Ведущий размер для B.
                &beta,          // Указатель на скаляр beta
                dummy_result,   // Матрица C (результат)
                N               // Ведущий размер для C.
    );
}

std::vector<MKL_Complex16 *> CreateBasisArray(int N) {
    // 2. В цикле выделяем память для каждой матрицы и добавляем указатель в
    // вектор
    std::vector<MKL_Complex16 *> basis_array;
    for (int j = 0; j < N; ++j) {
        for (int k = j + 1; k < N; ++k) {
            MKL_Complex16 *new_matrix =
                (MKL_Complex16 *)mkl_malloc(N * N * sizeof(MKL_Complex16), 64);
            memset(new_matrix, 0, N * N * sizeof(MKL_Complex16));
            if (new_matrix == NULL) {
                std::cerr << "Ошибка выделения памяти для матрицы";
                // Освобождаем все, что успели выделить
                for (MKL_Complex16 *mat : basis_array) {
                    mkl_free(mat);
                }
                // return 1;
            }

            // Инициализация

            int index = j * N + k;
            new_matrix[index] = {1. / sqrt(2), 0.};  // другой способ инициализации
            index = k * N + j;
            new_matrix[index] = {1. / sqrt(2), 0.};  // другой способ инициализации

            // Добавляем указатель на созданную матрицу в наш вектор
            basis_array.push_back(new_matrix);
        }
    }

    for (int j = 0; j < N; ++j) {
        for (int k = j + 1; k < N; ++k) {
            MKL_Complex16 *new_matrix =
                (MKL_Complex16 *)mkl_malloc(N * N * sizeof(MKL_Complex16), 64);
            memset(new_matrix, 0, N * N * sizeof(MKL_Complex16));
            if (new_matrix == NULL) {
                std::cerr << "Ошибка выделения памяти для матрицы";
                // Освобождаем все, что успели выделить
                for (MKL_Complex16 *mat : basis_array) {
                    mkl_free(mat);
                }
                // return 1;
            }

            // Инициализация

            int index = j * N + k;
            new_matrix[index] = {0., -1. / sqrt(2)};  // другой способ инициализации
            index = k * N + j;
            new_matrix[index] = {0., 1. / sqrt(2)};  // другой способ инициализации

            // Добавляем указатель на созданную матрицу в наш вектор
            basis_array.push_back(new_matrix);
        }
    }

    for (int l = 0; l < N - 1; ++l) {
        MKL_Complex16 *new_matrix = (MKL_Complex16 *)mkl_malloc(N * N * sizeof(MKL_Complex16), 64);
        memset(new_matrix, 0, N * N * sizeof(MKL_Complex16));
        if (new_matrix == NULL) {
            std::cerr << "Ошибка выделения памяти для матрицы";
            // Освобождаем все, что успели выделить
            for (MKL_Complex16 *mat : basis_array) {
                mkl_free(mat);
            }
            // return 1;
        }

        // Инициализация

        for (int k = 0; k < l + 1; ++k) {
            int index = k * N + k;
            new_matrix[index] = {1. / sqrt((l + 1) * (l + 2)), 0.};
        }
        int index = (l + 1) * N + (l + 1);
        new_matrix[index] = {-sqrt(l + 1) / sqrt(l + 2), 0.};
        basis_array.push_back(new_matrix);
    }

    return basis_array;
}

size_t Mapping(int i, int j, int N) {
    return i * (N - 1) - (i * (i - 1)) / 2 + j - i - 1;
}

bool Check(MKL_Complex16 *first, MKL_Complex16 *second, int N) {
    for (int i = 0; i < N * N; ++i) {
        if (abs(first[i].real - second[i].real) > 1e-4 or
            abs(first[i].imag - second[i].imag) > 1e-4) {
            return false;
        }
    }
    return true;
}

int main() {
    // Параметры
    int N = 9;
    int M = N * N - 1;

    std::vector<MKL_Complex16 *> basis_array = CreateBasisArray(N);

    std::vector<std::vector<MKL_Complex16 *>> commutator(M);
    std::vector<std::vector<MKL_Complex16 *>> new_commutator(M);
    for (int m = 0; m < M; ++m) {
        commutator[m].resize(M);
        new_commutator[m].resize(M);
        for (int n = 0; n < M; ++n) {
            commutator[m][n] = (MKL_Complex16 *)mkl_malloc(N * N * sizeof(MKL_Complex16), 64);
            new_commutator[m][n] = (MKL_Complex16 *)mkl_malloc(N * N * sizeof(MKL_Complex16), 64);
            Commutator(basis_array[m], basis_array[n], commutator[m][n], N);
        }
    }

    // 1 квадрант
    for (int j_k = 1; j_k + 1 < N; ++j_k) {
        for (int i = 0; i < j_k; ++i) {
            for (int l = j_k + 1; l < N; ++l) {
                size_t f_c = Mapping(i, j_k, N);
                size_t s_c = Mapping(j_k, l, N);
                new_commutator[f_c][s_c][i * N + l] = {0.5, 0};
                new_commutator[f_c][s_c][l * N + i] = {-0.5, 0};
            }
        }
    }

    for (int i_l = 1; i_l + 1 < N; ++i_l) {
        for (int j = i_l + 1; j < N; ++j) {
            for (int k = 0; k < i_l; ++k) {
                size_t f_c = Mapping(i_l, j, N);
                size_t s_c = Mapping(k, i_l, N);

                new_commutator[f_c][s_c][k * N + j] = {-0.5, 0};
                new_commutator[f_c][s_c][j * N + k] = {0.5, 0};
            }
        }
    }

    for (int i_k = 0; i_k + 1 < N; ++i_k) {
        for (int j = i_k + 1; j < N; ++j) {
            for (int l = i_k + 1; l < N; ++l) {
                if (j != l) {
                    size_t f_c = Mapping(i_k, j, N);
                    size_t s_c = Mapping(i_k, l, N);

                    new_commutator[f_c][s_c][j * N + l] = {0.5, 0};
                    new_commutator[f_c][s_c][l * N + j] = {-0.5, 0};
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

                    new_commutator[f_c][s_c][i * N + k] = {0.5, 0};
                    new_commutator[f_c][s_c][k * N + i] = {-0.5, 0};
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
                size_t s_c = N * (N-1) / 2 + Mapping(j_k, l, N);
                new_commutator[f_c][s_c][i * N + l] = {0, -0.5};
                new_commutator[f_c][s_c][l * N + i] = {0, -0.5};


                new_commutator[s_c][f_c][i * N + l] = {0, 0.5};
                new_commutator[s_c][f_c][l * N + i] = {0, 0.5};
            }
        }
    }

    // k < j
    for (int i_l = 1; i_l + 1 < N; ++i_l) {
        for (int j = i_l + 1; j < N; ++j) {
            for (int k = 0; k < i_l; ++k) {
                size_t f_c = Mapping(i_l, j, N);
                size_t s_c = N * (N-1) / 2 + Mapping(k, i_l, N);

                new_commutator[f_c][s_c][k * N + j] = {0, 0.5};
                new_commutator[f_c][s_c][j * N + k] = {0, 0.5};


                new_commutator[s_c][f_c][k * N + j] = {0, -0.5};
                new_commutator[s_c][f_c][j * N + k] = {0, -0.5};
            }
        }
    }

    for (int i_k = 0; i_k + 1 < N; ++i_k) {
        for (int j = i_k + 1; j < N; ++j) {
            for (int l = i_k + 1; l < N; ++l) {
                if (j != l) {
                    size_t f_c = Mapping(i_k, j, N);
                    size_t s_c = N * (N-1) / 2 + Mapping(i_k, l, N);

                    new_commutator[f_c][s_c][j * N + l] = {0, -0.5};
                    new_commutator[f_c][s_c][l * N + j] = {0, -0.5};

                    new_commutator[s_c][f_c][j * N + l] = {0, 0.5};
                    new_commutator[s_c][f_c][l * N + j] = {0, 0.5};
                }
            }
        }
    }


    for (int j_l = 1; j_l < N; ++j_l) {
        for (int i = 0; i < j_l; ++i) {
            for (int k = 0; k < j_l; ++k) {
                if (i != k) {
                    size_t f_c = Mapping(i, j_l, N);
                    size_t s_c = N * (N-1) / 2 + Mapping(k, j_l, N);

                    new_commutator[f_c][s_c][i * N + k] = {0, 0.5};
                    new_commutator[f_c][s_c][k * N + i] = {0, 0.5};

                    new_commutator[s_c][f_c][i * N + k] = {0, -0.5};
                    new_commutator[s_c][f_c][k * N + i] = {0, -0.5};
                }
            }
        }
    }

    for (int i_k = 0; i_k + 1 < N; ++i_k) {
        for (int j_l = i_k + 1; j_l < N; ++j_l) {
            size_t f_c = Mapping(i_k, j_l, N);
            size_t s_c = N * (N-1) / 2 + Mapping(i_k, j_l, N);

            new_commutator[f_c][s_c][i_k * N + i_k] = {0, 1.};
            new_commutator[f_c][s_c][j_l * N + j_l] = {0, -1.};

            new_commutator[s_c][f_c][i_k * N + i_k] = {0, -1.};
            new_commutator[s_c][f_c][j_l * N + j_l] = {0, 1.};
        }
    }



    // 3 квадрант и 7 квадрант
    for (int l = 0; l < N - 1; ++l) {

        // j = l + 1
        for (int i = 0; i < l + 1; ++i) {
            size_t f_c = Mapping(i, l + 1, N);
            size_t s_c = N * (N - 1) + l;
            new_commutator[f_c][s_c][i * N + l + 1] = {-sqrt(0.5 * (l + 2) / (l + 1)), 0};
            new_commutator[f_c][s_c][(l + 1) * N + i] = {sqrt(0.5 * (l + 2) / (l + 1)), 0};

            new_commutator[s_c][f_c][i * N + l + 1] = {sqrt(0.5 * (l + 2) / (l + 1)), 0};
            new_commutator[s_c][f_c][(l + 1) * N + i] = {-sqrt(0.5 * (l + 2) / (l + 1)), 0};
        }

        for (int i = 0; i < l + 1; ++i) {
            for (int j = l + 2; j < N; ++j) {
                size_t f_c = Mapping(i, j, N);
                size_t s_c = N * (N - 1) + l;
                new_commutator[f_c][s_c][i * N + j] = {-1. / sqrt(2. * (l + 1) * (l + 2)), 0};
                new_commutator[f_c][s_c][j * N + i] = {1. / sqrt(2. * (l + 1) * (l + 2)), 0};


                new_commutator[s_c][f_c][i * N + j] = {1. / sqrt(2. * (l + 1) * (l + 2)), 0};
                new_commutator[s_c][f_c][j * N + i] = {-1. / sqrt(2. * (l + 1) * (l + 2)), 0};
            }
        }

        // i = l + 1
        for (int j = l + 2; j < N; ++j) {
            size_t f_c = Mapping(l + 1, j, N);
            size_t s_c = N * (N - 1) + l;
            new_commutator[f_c][s_c][(l + 1) * N + j] = {sqrt(0.5 * (l + 1) / (l + 2)), 0};
            new_commutator[f_c][s_c][j * N + (l + 1)] = {-sqrt(0.5 * (l + 1) / (l + 2)), 0};

            new_commutator[s_c][f_c][(l + 1) * N + j] = {-sqrt(0.5 * (l + 1) / (l + 2)), 0};
            new_commutator[s_c][f_c][j * N + (l + 1)] = {sqrt(0.5 * (l + 1) / (l + 2)), 0};
        }
    }


    // 5 квадрант
    for (int j_k = 1; j_k + 1 < N; ++j_k) {
        for (int i = 0; i < j_k; ++i) {
            for (int l = j_k + 1; l < N; ++l) {
                size_t f_c = N * (N - 1) / 2 + Mapping(i, j_k, N);
                size_t s_c = N * (N - 1) / 2 + Mapping(j_k, l, N);
                new_commutator[f_c][s_c][i * N + l] = {-0.5, 0};
                new_commutator[f_c][s_c][l * N + i] = {0.5, 0};
            }
        }
    }

    for (int i_l = 1; i_l + 1 < N; ++i_l) {
        for (int j = i_l + 1; j < N; ++j) {
            for (int k = 0; k < i_l; ++k) {
                size_t f_c = N * (N - 1) / 2 + Mapping(i_l, j, N);
                size_t s_c = N * (N - 1) / 2 + Mapping(k, i_l, N);

                new_commutator[f_c][s_c][k * N + j] = {0.5, 0};
                new_commutator[f_c][s_c][j * N + k] = {-0.5, 0};
            }
        }
    }

    for (int i_k = 0; i_k + 1 < N; ++i_k) {
        for (int j = i_k + 1; j < N; ++j) {
            for (int l = i_k + 1; l < N; ++l) {
                if (j != l) {
                    size_t f_c = N * (N - 1) / 2 + Mapping(i_k, j, N);
                    size_t s_c = N * (N - 1) / 2 + Mapping(i_k, l, N);

                    new_commutator[f_c][s_c][j * N + l] = {0.5, 0};
                    new_commutator[f_c][s_c][l * N + j] = {-0.5, 0};
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

                    new_commutator[f_c][s_c][i * N + k] = {0.5, 0};
                    new_commutator[f_c][s_c][k * N + i] = {-0.5, 0};
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
            new_commutator[f_c][s_c][i * N + l + 1] = {0, sqrt(0.5 * (l + 2) / (l + 1))};
            new_commutator[f_c][s_c][(l + 1) * N + i] = {0, sqrt(0.5 * (l + 2) / (l + 1))};

            new_commutator[s_c][f_c][i * N + l + 1] = {0, -sqrt(0.5 * (l + 2) / (l + 1))};
            new_commutator[s_c][f_c][(l + 1) * N + i] = {0, -sqrt(0.5 * (l + 2) / (l + 1))};
        }

        for (int i = 0; i < l + 1; ++i) {
            for (int j = l + 2; j < N; ++j) {
                size_t f_c = N * (N - 1) / 2 + Mapping(i, j, N);
                size_t s_c = N * (N - 1) + l;
                new_commutator[f_c][s_c][i * N + j] = {0, 1. / sqrt(2. * (l + 1) * (l + 2))};
                new_commutator[f_c][s_c][j * N + i] = {0, 1. / sqrt(2. * (l + 1) * (l + 2))};

                new_commutator[s_c][f_c][i * N + j] = {0, -1. / sqrt(2. * (l + 1) * (l + 2))};
                new_commutator[s_c][f_c][j * N + i] = {0, -1. / sqrt(2. * (l + 1) * (l + 2))};
            }
        }

        // i = l + 1
        for (int j = l + 2; j < N; ++j) {
            size_t f_c = N * (N - 1) / 2 + Mapping(l + 1, j, N);
            size_t s_c = N * (N - 1) + l;
            new_commutator[f_c][s_c][(l + 1) * N + j] = {0, -sqrt(0.5 * (l + 1) / (l + 2))};
            new_commutator[f_c][s_c][j * N + (l + 1)] = {0, -sqrt(0.5 * (l + 1) / (l + 2))};

            new_commutator[s_c][f_c][(l + 1) * N + j] = {0, sqrt(0.5 * (l + 1) / (l + 2))};
            new_commutator[s_c][f_c][j * N + (l + 1)] = {0, sqrt(0.5 * (l + 1) / (l + 2))};
        }
    }


    



    for (int m = 0; m < N * N - 1; ++m) {
        for (int n = 0; n < N * N - 1; ++n) {
            if (not Check(commutator[m][n], new_commutator[m][n], N)) {
                std::cout << m << " " << n << "\n";
            }
        }
    }
    // std::cout << Check(commutator[0][1], new_commutator[0][1], N);

    // print_matrix_rowmajor(commutator[2][13], N, "one");
    // print_matrix_rowmajor(new_commutator[2][13], N, "two");
}