#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <cstring>
#include "mkl.h"
#include "generate_matrices.h"

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

// Helper: allocate and check
static void *xmalloc(size_t n) {
    void *p = mkl_malloc(n, 64);
    if (!p) {
        fprintf(stderr, "Allocation failed\n");
        exit(1);
    }
    return p;
}

// Compute conjugate transpose of A (N x N, row-major) into A_dag
// A and A_dag are MKL_Complex16 arrays of length N*N

// ----------------------------------------------------------------------
// Helper: conjugate transpose (row-major)
void conj_transpose(int N, const MKL_Complex16 *A, MKL_Complex16 *A_dag) {
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j) {
            MKL_Complex16 a = A[i*N + j];
            MKL_Complex16 c; c.real = a.real; c.imag = -a.imag;
            A_dag[j*N + i] = c;
        }
}

// ----------------------------------------------------------------------
// compute -i [H, rho]  (row-major)
// out must be preallocated N*N
void compute_commutator(int N,
                        const MKL_Complex16 *H,
                        const MKL_Complex16 *rho,
                        MKL_Complex16 *out)
{
    MKL_Complex16 alpha, beta;
    alpha.real = 1.0; alpha.imag = 0.0;
    beta.real = 0.0;  beta.imag = 0.0;

    MKL_Complex16 *A = (MKL_Complex16*)xmalloc(sizeof(MKL_Complex16)*N*N); // H * rho
    MKL_Complex16 *B = (MKL_Complex16*)xmalloc(sizeof(MKL_Complex16)*N*N); // rho * H

    // A = H * rho
    cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                N, N, N,
                &alpha, H, N, rho, N, &beta, A, N);

    // B = rho * H
    cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                N, N, N,
                &alpha, rho, N, H, N, &beta, B, N);

    // Comm = A - B  (complex subtraction)
    for (int k = 0; k < N*N; ++k) {
        double ar=A[k].real, ai=A[k].imag;
        double br=B[k].real, bi=B[k].imag;
        double cr = ar - br; double ci = ai - bi;
        // multiply by -i: (-i)*(cr + i ci) = ci - i*cr
        out[k].real = ci;
        out[k].imag = -cr;
    }

    mkl_free(A);
    mkl_free(B);
}

// ----------------------------------------------------------------------
// compute Ld = L rho L^dag - 1/2 (L^dag L rho + rho L^dag L)
// uses temporaries, all row-major
void compute_Ld(int N,
                const MKL_Complex16 *L,
                const MKL_Complex16 *rho,
                MKL_Complex16 *Ld)
{
    MKL_Complex16 alpha, beta;
    alpha.real = 1.0; alpha.imag = 0.0;
    beta.real = 0.0;  beta.imag = 0.0;

    MKL_Complex16 *Ldag  = (MKL_Complex16*)xmalloc(sizeof(MKL_Complex16)*N*N);
    MKL_Complex16 *T1    = (MKL_Complex16*)xmalloc(sizeof(MKL_Complex16)*N*N); // L * rho
    MKL_Complex16 *T2    = (MKL_Complex16*)xmalloc(sizeof(MKL_Complex16)*N*N); // L rho Ldag
    MKL_Complex16 *M     = (MKL_Complex16*)xmalloc(sizeof(MKL_Complex16)*N*N); // Ldag * L
    MKL_Complex16 *T3    = (MKL_Complex16*)xmalloc(sizeof(MKL_Complex16)*N*N); // M * rho
    MKL_Complex16 *T4    = (MKL_Complex16*)xmalloc(sizeof(MKL_Complex16)*N*N); // rho * M

    conj_transpose(N, L, Ldag);

    // T1 = L * rho
    cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, N,N,N, &alpha, L, N, rho, N, &beta, T1, N);
    // T2 = T1 * Ldag = L rho Ldag
    cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, N,N,N, &alpha, T1, N, Ldag, N, &beta, T2, N);
    // M = Ldag * L
    cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, N,N,N, &alpha, Ldag, N, L, N, &beta, M, N);
    // T3 = M * rho
    cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, N,N,N, &alpha, M, N, rho, N, &beta, T3, N);
    // T4 = rho * M
    cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, N,N,N, &alpha, rho, N, M, N, &beta, T4, N);

    // Ld = T2 - 0.5*(T3 + T4)
    // copy T2 -> Ld
    memcpy(Ld, T2, sizeof(MKL_Complex16)*N*N);
    // T3 += T4
    for (int k = 0; k < N*N; ++k) {
        T3[k].real += T4[k].real;
        T3[k].imag += T4[k].imag;
    }
    // Ld -= 0.5 * T3
    for (int k = 0; k < N*N; ++k) {
        double tr = 0.5 * T3[k].real;
        double ti = 0.5 * T3[k].imag;
        Ld[k].real -= tr;
        Ld[k].imag -= ti;
    }

    mkl_free(Ldag);
    mkl_free(T1);
    mkl_free(T2);
    mkl_free(M);
    mkl_free(T3);
    mkl_free(T4);
}

// ----------------------------------------------------------------------
// total Liouvillian L(rho) = -i[H,rho] + Ld
void L_of_rho(int N,
              const MKL_Complex16 *H,
              const MKL_Complex16 *L,
              const MKL_Complex16 *rho,
              MKL_Complex16 *out)
{
    MKL_Complex16 *comm = (MKL_Complex16*)xmalloc(sizeof(MKL_Complex16)*N*N);
    MKL_Complex16 *Ld   = (MKL_Complex16*)xmalloc(sizeof(MKL_Complex16)*N*N);

    compute_commutator(N, H, rho, comm);
    compute_Ld(N, L, rho, Ld);

    // out = comm + Ld
    for (int k = 0; k < N*N; ++k) {
        out[k].real = comm[k].real + Ld[k].real;
        out[k].imag = comm[k].imag + Ld[k].imag;
    }

    mkl_free(comm);
    mkl_free(Ld);
}

// ----------------------------------------------------------------------
// helper: out = a + b * scalar (out = a + coeff * b), length N*N
void add_scaled(int nn, const MKL_Complex16 *a, const MKL_Complex16 *b, double coeff, MKL_Complex16 *out)
{
    for (int k = 0; k < nn; ++k) {
        out[k].real = a[k].real + coeff * b[k].real;
        out[k].imag = a[k].imag + coeff * b[k].imag;
    }
}

// ----------------------------------------------------------------------
// RK4 integrator step: rho_{n+1} = rho_n + dt/6*(k1 + 2k2 + 2k3 + k4)
// where k1 = L(rho_n), k2 = L(rho_n + dt/2*k1), ...
void rk4_step(int N,
              const MKL_Complex16 *H,
              const MKL_Complex16 *L,
              const MKL_Complex16 *rho_in,
              double dt,
              MKL_Complex16 *rho_out)
{
    int nn = N*N;
    MKL_Complex16 *k1 = (MKL_Complex16*)xmalloc(sizeof(MKL_Complex16)*nn);
    MKL_Complex16 *k2 = (MKL_Complex16*)xmalloc(sizeof(MKL_Complex16)*nn);
    MKL_Complex16 *k3 = (MKL_Complex16*)xmalloc(sizeof(MKL_Complex16)*nn);
    MKL_Complex16 *k4 = (MKL_Complex16*)xmalloc(sizeof(MKL_Complex16)*nn);
    MKL_Complex16 *temp = (MKL_Complex16*)xmalloc(sizeof(MKL_Complex16)*nn);

    // k1 = L(rho_in)
    L_of_rho(N, H, L, rho_in, k1);

    // temp = rho_in + dt/2 * k1
    add_scaled(nn, rho_in, k1, dt*0.5, temp);
    L_of_rho(N, H, L, temp, k2);

    // temp = rho_in + dt/2 * k2
    add_scaled(nn, rho_in, k2, dt*0.5, temp);
    L_of_rho(N, H, L, temp, k3);

    // temp = rho_in + dt * k3
    add_scaled(nn, rho_in, k3, dt, temp);
    L_of_rho(N, H, L, temp, k4);

    // rho_out = rho_in + dt/6 * (k1 + 2k2 + 2k3 + k4)
    for (int k = 0; k < nn; ++k) {
        double real_sum = k1[k].real + 2.0*k2[k].real + 2.0*k3[k].real + k4[k].real;
        double imag_sum = k1[k].imag + 2.0*k2[k].imag + 2.0*k3[k].imag + k4[k].imag;
        rho_out[k].real = rho_in[k].real + (dt/6.0)*real_sum;
        rho_out[k].imag = rho_in[k].imag + (dt/6.0)*imag_sum;
    }

    mkl_free(k1); mkl_free(k2); mkl_free(k3); mkl_free(k4); mkl_free(temp);
}

// ----------------------------------------------------------------------
// small utility: print matrix
void print_mat(int N, const MKL_Complex16 *A, const char *name) {
    printf("%s =\n", name);
    for (int i=0;i<N;i++) {
        for (int j=0;j<N;j++) {
            MKL_Complex16 x = A[i*N+j];
            printf("(% .6f,% .6f)  ", x.real, x.imag);
        }
        printf("\n");
    }
}


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
    int N = 4;
    int M = N * N - 1;

    // создаем гамильтониан и заполняем его нулями
    MKL_Complex16 *hamiltonian;
    GenerateTracelessHamiltonian(N, 2, hamiltonian);

    // создаем линдбладиан и заполняем его нулями
    MKL_Complex16 *lindbladian;
    GenerateLp(N, 2, lindbladian);

    MKL_Complex16 *rho;
    GenerateDensity(N, 2, rho);

    std::vector<MKL_Complex16 *> basis_array = CreateBasisArray(N);

    double *v = GetVCoef(basis_array, rho, N);
    for (int i = 0; i < N * N - 1; ++i) {
        std::cout << v[i] << " ";
    }
    std::cout << "\n";

    MKL_Complex16 *rho_next = (MKL_Complex16*)xmalloc(sizeof(MKL_Complex16)*N*N);

    

    // Параметры
    double t_end = 1.3;
    double h = 0.01;

    double t0 = 0.0;

    double t = t0;
    std::cout << std::fixed << std::setprecision(6);
    // double *workspace_A = (double*)mkl_malloc(M * M * sizeof(double), 64);
    while (t < t_end + h / 2) {

        rk4_step(N, hamiltonian, lindbladian, rho, h, rho_next);
        // swap rho and rho_next
        memcpy(rho, rho_next, sizeof(MKL_Complex16)*N*N);

        printf("t = %.3f\n", t);
        // print_mat(N, rho, "rho");

        double *v = GetVCoef(basis_array, rho, N);
        for (int i = 0; i < N * N - 1; ++i) {
            std::cout << v[i] << " ";
        }
        std::cout << "\n";

        t += h;
    }

    print_mat(N, rho, "rho");

    // print_matrix_rowmajor(matrix_rho, N, "result");
    MKL_Complex16 tr = Trace(rho, N);


    std::cout << std::fixed << std::setprecision(17) << tr.real - 1 << " " << tr.imag << "\n";
    std::cout << tr.real - 1 << "\n";

    std::cout << std::fixed << std::setprecision(17) << check_hermitian_approx(rho, N)
              << "\n";



    mkl_free(rho_next);
   
    mkl_free(hamiltonian);
    mkl_free(lindbladian);
    mkl_free(rho);

    return 0;
}
