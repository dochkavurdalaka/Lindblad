#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <cstring>
#include <chrono>
#include "mkl.h"
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

MKL_Complex16 operator*(const MKL_Complex16 &a, const MKL_Complex16 &b) {
    MKL_Complex16 result;
    result.real = a.real * b.real - a.imag * b.imag;
    result.imag = a.real * b.imag + a.imag * b.real;
    return result;
}

MKL_Complex16 operator+(const MKL_Complex16 &a, const MKL_Complex16 &b) {
    MKL_Complex16 result;
    result.real = a.real + b.real;
    result.imag = a.imag + b.imag;
    return result;
}

MKL_Complex16 operator+=(MKL_Complex16 &a, const MKL_Complex16 &b) {
    a.real += b.real;
    a.imag += b.imag;
    return a;
}

MKL_Complex16 operator*=(MKL_Complex16 &a, const MKL_Complex16 &b) {
    double real = a.real;
    a.real = real * b.real - a.imag * b.imag;
    a.imag = real * b.imag + a.imag * b.real;
    return a;
}

MKL_Complex16 operator*(double a, const MKL_Complex16 &b) {
    MKL_Complex16 result;
    result.real = a * b.real;
    result.imag = a * b.imag;
    return result;
}

MKL_Complex16 operator*(const MKL_Complex16 &a, double b) {
    MKL_Complex16 result;
    result.real = b * a.real;
    result.imag = b * a.imag;
    return result;
}

// здесь и далее попробуем решить систему для константного H
// тогда матрица Q тоже будет константной

// Вспомогательная функция для печати вектора
void print_vector(const char *name, double t, const double *v, int M) {
    std::cout << name << "(t=" << t << "): [ ";
    for (int i = 0; i < M; ++i) {
        std::cout << v[i] << ", ";
    }
    std::cout << "]" << std::endl;
}

// Реализация нашей векторной функции f(t, v) = (Q(t) + R)v + K
// result = (Q(t) + R)v + K
void calculate_f(double t, const double *v, double *result, const double *Q, const double *R,
                 const double *K, double *workspace_A, int M) {
    double *A = workspace_A;

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

std::vector<double> GetHCoef(const std::vector<MKL_Complex16 *> &basis_array,
                             MKL_Complex16 *hamiltonian, int N) {
    MKL_Complex16 alpha = {1.0, 0.0}, beta = {0.0, 0.0};
    MKL_Complex16 *result_matrix = (MKL_Complex16 *)mkl_malloc(N * N * sizeof(MKL_Complex16), 64);
    std::vector<double> h_coeff;
    for (MKL_Complex16 *mat : basis_array) {
        cblas_zgemm(CblasRowMajor,  // Указывает, что матрицы хранятся построчно
                                    // (стандарт для C/C++)
                    CblasNoTrans,   // Операция для A: не транспонировать
                    CblasNoTrans,   // Операция для B: не транспонировать
                    N,              // Количество строк в матрице A (и C)
                    N,              // Количество столбцов в матрице B (и C)
                    N,              // Количество столбцов в A и строк в B
                    &alpha,         // Указатель на скаляр alpha
                    hamiltonian,    // Матрица A
                    N,              // Ведущий размер (leading dimension) для A. Для RowMajor это
                                    // количество столбцов.
                    mat,            // Матрица B
                    N,              // Ведущий размер для B.
                    &beta,          // Указатель на скаляр beta
                    result_matrix,  // Матрица C (результат)
                    N               // Ведущий размер для C.
        );
        h_coeff.push_back(Trace(result_matrix, N).real);
    }
    return h_coeff;
}

std::vector<MKL_Complex16> GetLCoef(const std::vector<MKL_Complex16 *> &basis_array,
                                    MKL_Complex16 *lindbladian, int N) {
    MKL_Complex16 alpha = {1.0, 0.0}, beta = {0.0, 0.0};
    MKL_Complex16 *result_matrix = (MKL_Complex16 *)mkl_malloc(N * N * sizeof(MKL_Complex16), 64);
    std::vector<MKL_Complex16> l_coeff;

    for (MKL_Complex16 *mat : basis_array) {
        cblas_zgemm(CblasRowMajor,  // Указывает, что матрицы хранятся построчно
                                    // (стандарт для C/C++)
                    CblasNoTrans,   // Операция для A: не транспонировать
                    CblasNoTrans,   // Операция для B: не транспонировать
                    N,              // Количество строк в матрице A (и C)
                    N,              // Количество столбцов в матрице B (и C)
                    N,              // Количество столбцов в A и строк в B
                    &alpha,         // Указатель на скаляр alpha
                    lindbladian,    // Матрица A
                    N,              // Ведущий размер (leading dimension) для A. Для RowMajor это
                                    // количество столбцов.
                    mat,            // Матрица B
                    N,              // Ведущий размер для B.
                    &beta,          // Указатель на скаляр beta
                    result_matrix,  // Матрица C (результат)
                    N               // Ведущий размер для C.
        );
        l_coeff.push_back(Trace(result_matrix, N));
    }
    return l_coeff;
}

double *GetVCoef(const std::vector<MKL_Complex16 *> &basis_array, MKL_Complex16 *rho, int N) {
    MKL_Complex16 alpha = {1.0, 0.0}, beta = {0.0, 0.0};
    MKL_Complex16 *result_matrix = (MKL_Complex16 *)mkl_malloc(N * N * sizeof(MKL_Complex16), 64);
    double *v_coeff = (double *)mkl_malloc((N * N - 1) * sizeof(double), 64);
    size_t ind = 0;
    for (MKL_Complex16 *mat : basis_array) {
        // print_matrix_rowmajor(mat, 2, "f");
        cblas_zgemm(CblasRowMajor,  // Указывает, что матрицы хранятся построчно
                                    // (стандарт для C/C++)
                    CblasNoTrans,   // Операция для A: не транспонировать
                    CblasNoTrans,   // Операция для B: не транспонировать
                    N,              // Количество строк в матрице A (и C)
                    N,              // Количество столбцов в матрице B (и C)
                    N,              // Количество столбцов в A и строк в B
                    &alpha,         // Указатель на скаляр alpha
                    rho,            // Матрица A
                    N,              // Ведущий размер (leading dimension) для A. Для RowMajor это
                                    // количество столбцов.
                    mat,            // Матрица B
                    N,              // Ведущий размер для B.
                    &beta,          // Указатель на скаляр beta
                    result_matrix,  // Матрица C (результат)
                    N               // Ведущий размер для C.
        );
        v_coeff[ind] = Trace(result_matrix, N).real;
        ++ind;
    }
    return v_coeff;
}

// Проверка приближенной эрмитовости: M == M^† в пределах eps
double check_hermitian_approx(const MKL_Complex16 *M, int d) {
    double result = 0.0;
    for (int i = 0; i < d; ++i) {
        for (int j = 0; j < d; ++j) {
            const MKL_Complex16 &a = M[i * d + j];
            const MKL_Complex16 &b = M[j * d + i];  // should be conj(a)
            double re_diff = a.real - b.real;
            double im_diff = a.imag + b.imag;  // a.imag - (-b.imag) since b should be conj(a)
            result = std::max(re_diff * re_diff + im_diff * im_diff, result);
        }
    }
    return result;
}

int main() {
    // Параметры
    int N = 5;
    int M = N * N - 1;

    // создаем гамильтониан и заполняем его нулями
    MKL_Complex16 *hamiltonian;
    GenerateTracelessHamiltonian(N, 2, hamiltonian);

    // создаем линдбладиан и заполняем его нулями
    MKL_Complex16 *lindbladian;
    GenerateLp(N, 2, lindbladian);

    MKL_Complex16 *rho;
    GenerateDensity(N, 2, rho);

    // вот здесь нужно будет проинициализировать вектор гамма
    int P = 1;
    std::vector<MKL_Complex16> gamma_coeff(P);
    for (int i = 0; i < P; ++i) {
        gamma_coeff[i].real = 1;
        gamma_coeff[i].imag = 0;
    }

    // 1. Создаем вектор, который будет хранить указатели на наши матрицы
    std::vector<MKL_Complex16 *> basis_array = CreateBasisArray(N);
    double *v = GetVCoef(basis_array, rho, N);
    // for (int i = 0; i < N * N - 1; ++i) {
    //     std::cout << v[i] << " ";
    // }

    auto start = high_resolution_clock::now();

    // вычисляем коэффициенты h
    MKL_Complex16 alpha = {1.0, 0.0}, beta = {0.0, 0.0};
    std::vector<double> h_coeff = GetHCoef(basis_array, hamiltonian, N);
        std::cout << "h_coeff\n";
    for(auto el: h_coeff) {
        std::cout << el << " ";
    }
    std::cout << "\n";
    // вычисляем коэффициенты l
    std::vector<MKL_Complex16> l_coeff = GetLCoef(basis_array, lindbladian, N);
    
    std::cout << "l_coeff:\n";
    for(auto el: l_coeff) {
        std::cout << el.real << " " << el.imag << " ";
    }

    // Вычисляем разницу в секундах (как double)
    std::cout << "Коэффициенты l и h: "
              << duration_cast<duration<double>>(high_resolution_clock::now() - start).count()
              << "\n";
    start = high_resolution_clock::now();

    // Общее количество элементов
    size_t total_elements = M * M * M;

    // Выделение памяти для тензора
    double *f_tensor = (double *)mkl_malloc(total_elements * sizeof(double), 64);

    if (f_tensor == NULL) {
        std::cerr << "Ошибка выделения памяти для тензора." << std::endl;
        return 1;
    }

    double *d_tensor = (double *)mkl_malloc(total_elements * sizeof(double), 64);

    if (d_tensor == NULL) {
        std::cerr << "Ошибка выделения памяти для тензора." << std::endl;
        return 1;
    }

    MKL_Complex16 *z_tensor =
        (MKL_Complex16 *)mkl_malloc(total_elements * sizeof(MKL_Complex16), 64);

    if (z_tensor == NULL) {
        std::cerr << "Ошибка выделения памяти для тензора." << std::endl;
        return 1;
    }

    MKL_Complex16 *end_result = (MKL_Complex16 *)mkl_malloc(N * N * sizeof(MKL_Complex16), 64);

    std::vector<std::vector<MKL_Complex16 *>> commutator(M);
    std::vector<std::vector<MKL_Complex16 *>> anticommutator(M);
    for (int m = 0; m < M; ++m) {
        commutator[m].resize(M);
        anticommutator[m].resize(M);
        for (int n = 0; n < M; ++n) {
            commutator[m][n] = (MKL_Complex16 *)mkl_malloc(N * N * sizeof(MKL_Complex16), 64);
            Commutator(basis_array[m], basis_array[n], commutator[m][n], N);
            anticommutator[m][n] = (MKL_Complex16 *)mkl_malloc(N * N * sizeof(MKL_Complex16), 64);
            AntiCommutator(basis_array[m], basis_array[n], anticommutator[m][n], N);
        }
    }

    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < M; ++n) {
            for (int s = 0; s < M; ++s) {
                // Вычисляем линейный индекс
                int index = m * M * M + n * M + s;

                cblas_zgemm(CblasRowMajor,   // Указывает, что матрицы хранятся построчно
                                             // (стандарт для C/C++)
                            CblasNoTrans,    // Операция для A: не транспонировать
                            CblasNoTrans,    // Операция для B: не транспонировать
                            N,               // Количество строк в матрице A (и C)
                            N,               // Количество столбцов в матрице B (и C)
                            N,               // Количество столбцов в A и строк в B
                            &alpha,          // Указатель на скаляр alpha
                            basis_array[s],  // Матрица A
                            N,  // Ведущий размер (leading dimension) для A. Для RowMajor
                                // это количество столбцов.
                            commutator[m][n],  // Матрица B
                            N,                 // Ведущий размер для B.
                            &beta,             // Указатель на скаляр beta
                            end_result,        // Матрица C (результат)
                            N                  // Ведущий размер для C.
                );

                f_tensor[index] = Trace(end_result, N).imag;
                // std::cout << "m: " << m << ", n: " << n << ", s: " << s << "::::" <<
                // f_tensor[index] << "\n"; f_tensor[index] =
                // multiply_complex({0,-1},Trace(end_result, N));

                cblas_zgemm(CblasRowMajor,   // Указывает, что матрицы хранятся построчно
                                             // (стандарт для C/C++)
                            CblasNoTrans,    // Операция для A: не транспонировать
                            CblasNoTrans,    // Операция для B: не транспонировать
                            N,               // Количество строк в матрице A (и C)
                            N,               // Количество столбцов в матрице B (и C)
                            N,               // Количество столбцов в A и строк в B
                            &alpha,          // Указатель на скаляр alpha
                            basis_array[s],  // Матрица A
                            N,  // Ведущий размер (leading dimension) для A. Для RowMajor
                                // это количество столбцов.
                            anticommutator[m][n],  // Матрица B
                            N,                     // Ведущий размер для B.
                            &beta,                 // Указатель на скаляр beta
                            end_result,            // Матрица C (результат)
                            N                      // Ведущий размер для C.
                );

                d_tensor[index] = Trace(end_result, N).real;
                // std::cout << "m: " << m << ", n: " << n << ", s: " << s << "::::" <<
                // d_tensor[index] << "\n";

                z_tensor[index].real = f_tensor[index];
                z_tensor[index].imag = d_tensor[index];
            }
        }
    }
    // Вычисляем разницу в секундах (как double)
    std::cout << "Тензоры f, d, z: "
              << duration_cast<duration<double>>(high_resolution_clock::now() - start).count()
              << "\n";
    start = high_resolution_clock::now();

    double *q_tensor = (double *)mkl_malloc(M * M * sizeof(double), 64);

    if (q_tensor == NULL) {
        std::cerr << "Ошибка выделения памяти для тензора." << std::endl;
        return 1;
    }
    memset(q_tensor, 0, M * M * sizeof(double));

    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < M; ++n) {
            for (int s = 0; s < M; ++s) {
                int q_index = s * M + n;
                int f_index = m * M * M + n * M + s;

                q_tensor[q_index] += h_coeff[m] * f_tensor[f_index];
            }
        }
    }

    // Вычисляем разницу в секундах (как double)
    std::cout << "Матрица q: "
              << duration_cast<duration<double>>(high_resolution_clock::now() - start).count()
              << "\n";
    start = high_resolution_clock::now();

    // for (int s = 0; s < M; ++s) {
    //     for (int n = 0; n < M; ++n) {
    //         int q_index = s * M + n;
    //         std::cout << q_tensor[q_index] << " ";
    //     }
    //     std::cout << "\n";
    // }

    double *k_tensor = (double *)mkl_malloc(M * sizeof(double), 64);
    if (k_tensor == NULL) {
        std::cerr << "Ошибка выделения памяти для тензора." << std::endl;
        return 1;
    }
    memset(k_tensor, 0, M * sizeof(double));

    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < M; ++n) {
            for (int s = 0; s < M; ++s) {
                int f_index = m * M * M + n * M + s;
                k_tensor[s] += (l_coeff[m] * Conjugate(l_coeff[n]) * f_tensor[f_index]).imag;
            }
        }
    }

    for (int s = 0; s < M; ++s) {
        k_tensor[s] *= -1. / N;
        std::cout << k_tensor[s] << " ";
    }

    // Вычисляем разницу в секундах (как double)
    std::cout << "Вектор k: "
              << duration_cast<duration<double>>(high_resolution_clock::now() - start).count()
              << "\n";
    start = high_resolution_clock::now();

    double *r_tensor = (double *)mkl_malloc(M * M * sizeof(double), 64);
    if (r_tensor == NULL) {
        std::cerr << "Ошибка выделения памяти для тензора." << std::endl;
        return 1;
    }

    // Циклы для вычисления каждого элемента r_sn
    for (int s = 0; s < M; ++s) {
        for (int n = 0; n < M; ++n) {

            double sum_jkl = 0.;  // Сумма по j, k, l

            for (int j = 0; j < M; ++j) {
                for (int k = 0; k < M; ++k) {
                    for (int l = 0; l < M; ++l) {
                        // Вычисляем z_jln * f_kls
                        int z_ind = j * M * M + l * M + n;
                        int f_ind = k * M * M + l * M + s;
                        MKL_Complex16 term = z_tensor[z_ind] * f_tensor[f_ind];

                        // Вычисляем z_bar_kln * f_jls
                        z_ind = k * M * M + l * M + n;
                        f_ind = j * M * M + l * M + s;
                        term += Conjugate(z_tensor[z_ind]) * f_tensor[f_ind];

                        sum_jkl += (l_coeff[j] * Conjugate(l_coeff[k]) * term).real;
                    }
                }
            }
            r_tensor[s * M + n] = -0.25 * sum_jkl;
        }
    }

    print_double_matrix_rowmajor(r_tensor, M, "SSS");

    // Вычисляем разницу в секундах (как double)
    std::cout << "Матрица r: "
              << duration_cast<duration<double>>(high_resolution_clock::now() - start).count()
              << "\n";
    start = high_resolution_clock::now();

    // Начальные условия
    double t0 = 0.0;

    // Временные векторы для РК4
    double *k1 = (double *)mkl_malloc(M * sizeof(double), 64);
    double *k2 = (double *)mkl_malloc(M * sizeof(double), 64);
    double *k3 = (double *)mkl_malloc(M * sizeof(double), 64);
    double *k4 = (double *)mkl_malloc(M * sizeof(double), 64);
    double *v_temp = (double *)mkl_malloc(M * sizeof(double), 64);
    double *v_sum = (double *)mkl_malloc(M * sizeof(double), 64);

    // Параметры
    double t_end = 1.3;
    double h = 0.01;

    double t = t0;
    std::cout << std::fixed << std::setprecision(6);
    double *workspace_A = (double *)mkl_malloc(M * M * sizeof(double), 64);
    while (t < t_end + h / 2) {

        // --- Шаг метода Рунге-Кутты 4-го порядка ---

        // k1 = f(t, v)
        calculate_f(t, v, k1, q_tensor, r_tensor, k_tensor, workspace_A, M);

        // k2 = f(t + h/2, v + h/2 * k1)
        cblas_dcopy(M, v, 1, v_temp, 1);  // v_temp = v
        double alpha = 0.5 * h;
        cblas_daxpy(M, alpha, k1, 1, v_temp, 1);  // v_temp = v + 0.5*h*k1
        calculate_f(t + 0.5 * h, v_temp, k2, q_tensor, r_tensor, k_tensor, workspace_A, M);

        // k3 = f(t + h/2, v + h/2 * k2)
        cblas_dcopy(M, v, 1, v_temp, 1);          // v_temp = v
        cblas_daxpy(M, alpha, k2, 1, v_temp, 1);  // v_temp = v + 0.5*h*k2
        calculate_f(t + 0.5 * h, v_temp, k3, q_tensor, r_tensor, k_tensor, workspace_A, M);

        // k4 = f(t + h, v + h * k3)
        cblas_dcopy(M, v, 1, v_temp, 1);  // v_temp = v
        alpha = h;
        cblas_daxpy(M, alpha, k3, 1, v_temp, 1);  // v_temp = v + h*k3
        calculate_f(t + h, v_temp, k4, q_tensor, r_tensor, k_tensor, workspace_A, M);

        // Обновляем v: v = v + (h/6) * (k1 + 2k2 + 2k3 + k4)
        cblas_dcopy(M, k1, 1, v_sum, 1);  // v_sum = k1
        alpha = 2.0;
        cblas_daxpy(M, alpha, k2, 1, v_sum, 1);  // v_sum = k1 + 2*k2
        cblas_daxpy(M, alpha, k3, 1, v_sum, 1);  // v_sum = k1 + 2*k2 + 2*k3
        alpha = 1.0;
        cblas_daxpy(M, alpha, k4, 1, v_sum, 1);  // v_sum = k1 + 2*k2 + 2*k3 + k4
        alpha = h / 6.0;
        cblas_daxpy(M, alpha, v_sum, 1, v, 1);  // v = v + (h/6)*v_sum

        print_vector("v", t, v, M);

        t += h;
    }

    // Вычисляем разницу в секундах (как double)
    std::cout << "Интегрирование ДУ: "
              << duration_cast<duration<double>>(high_resolution_clock::now() - start).count()
              << "\n";

    MKL_Complex16 *matrix_rho = (MKL_Complex16 *)mkl_malloc(N * N * sizeof(MKL_Complex16), 64);
    memset(matrix_rho, 0, N * N * sizeof(MKL_Complex16));

    for (int i = 0; i < N * N; ++i) {
        for (int j = 0; j < M; ++j) {
            matrix_rho[i] += v[j] * basis_array[j][i];
        }
    }

    // print_matrix_rowmajor(matrix_rho, N, "result");
    MKL_Complex16 tr = Trace(matrix_rho, N);

    std::cout << std::fixed << std::setprecision(17) << tr.real << "\n";

    std::cout << std::fixed << std::setprecision(17) << check_hermitian_approx(matrix_rho, N)
              << "\n";

    // --- Очистка памяти ---
    // Вектор сам удалит массив указателей, но нам нужно вручную освободить
    // данные, на которые они указывают!
    for (MKL_Complex16 *mat_ptr : basis_array) {
        mkl_free(mat_ptr);
    }
    // mkl_free(middle_result);
    mkl_free(end_result);
    mkl_free(hamiltonian);
    mkl_free(lindbladian);
    mkl_free(rho);
    mkl_free(matrix_rho);
    mkl_free(f_tensor);
    mkl_free(d_tensor);
    mkl_free(z_tensor);
    mkl_free(q_tensor);
    mkl_free(k_tensor);
    mkl_free(r_tensor);

    // Освобождение памяти
    mkl_free(v);
    mkl_free(k1);
    mkl_free(k2);
    mkl_free(k3);
    mkl_free(k4);
    mkl_free(v_temp);
    mkl_free(v_sum);

    return 0;
}
