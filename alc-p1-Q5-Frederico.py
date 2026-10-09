import numpy as np


def resolve_lu(A, b):
    # Cópias em ponto flutuante para preservar A e b originais.
    U = np.array(A, dtype=float, copy=True)
    b_array = np.array(b, dtype=float, copy=True)

    # Verifica se A é uma matriz quadrada.
    if U.ndim != 2 or U.shape[0] != U.shape[1]:
        raise Exception("A deve ser uma matriz quadrada.")

    n = U.shape[0]

    if n == 0:
        raise Exception("A deve possuir pelo menos uma linha e uma coluna.")

    # Aceita b como vetor de dimensão n ou como vetor coluna n x 1.
    if b_array.ndim == 1 and b_array.shape[0] == n:
        b_vec = np.array(b_array, dtype=float, copy=True)

    elif (
        b_array.ndim == 2
        and b_array.shape[0] == n
        and b_array.shape[1] == 1
    ):
        b_vec = np.zeros(n, dtype=float)

        for i in range(n):
            b_vec[i] = b_array[i, 0]

    else:
        raise Exception("b deve ser um vetor com dimensão compatível com A.")

    # L começa como identidade: sua diagonal permanece igual a 1.
    L = np.eye(n, dtype=float)

    # Eliminação de Gauss sem pivoteamento.
    # U é atualizada e os multiplicadores da eliminação são armazenados em L.
    for j in range(n - 1):

        # O pivô deve ser verificado antes da divisão usada no multiplicador.
        if U[j, j] == 0.0:
            raise Exception(
                "Pivô nulo encontrado. "
                "Utilize uma função alternativa com pivoteamento."
            )

        for i in range(j + 1, n):

            # Multiplicador calculado com os valores atuais de U.
            m = U[i, j] / U[j, j]
            L[i, j] = m

            # Elimina o elemento abaixo do pivô.
            U[i, j] = 0.0

            for k in range(j + 1, n):
                U[i, k] = U[i, k] - m * U[j, k]

    # O último pivô não participa da eliminação,
    # mas será usado na substituição regressiva.
    if U[n - 1, n - 1] == 0.0:
        raise Exception(
            "Pivô nulo encontrado. "
            "Utilize uma função alternativa com pivoteamento."
        )

    # Substituição progressiva: resolve Ly = b.
    y = np.zeros(n, dtype=float)

    for i in range(n):
        soma = 0.0

        for j in range(i):
            soma = soma + L[i, j] * y[j]

        # Como a diagonal de L é igual a 1, não é necessário dividir.
        y[i] = b_vec[i] - soma

    # Substituição regressiva: resolve Ux = y.
    x = np.zeros(n, dtype=float)

    for i in range(n - 1, -1, -1):

        # Verificação imediatamente antes da divisão pelo pivô.
        if U[i, i] == 0.0:
            raise Exception(
                "Pivô nulo encontrado. "
                "Utilize uma função alternativa com pivoteamento."
            )

        soma = 0.0

        for j in range(i + 1, n):
            soma = soma + U[i, j] * x[j]

        x[i] = (y[i] - soma) / U[i, i]

    return L, U, x
