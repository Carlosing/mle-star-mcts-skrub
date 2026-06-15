# 1. Usar la versión exacta de Python
FROM python:3.12-slim

# 2. Instalar compiladores de C++ y herramientas del SO
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# 3. Instalar 'uv'
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# 4. Carpeta de trabajo
WORKDIR /app

# 5. Copiar configuración
COPY pyproject.toml uv.lock* README.md ./

# 5.5 EL HACK: Crear una carpeta falsa para engañar al empaquetador y salvar la caché
RUN mkdir machine_learning_engineering && touch machine_learning_engineering/__init__.py

# 6. Instalar dependencias pesadas aisladas del sistema
# --torch-backend=cpu asegura la rueda de torch solo-CPU en vez del build CUDA (~2 GB)
RUN uv pip install --system --torch-backend=cpu -e .

# 6.5 Herramientas de test (los tests de skrub corren dentro del contenedor)
RUN uv pip install --system pytest pytest-asyncio

# 7. Copiar el código real (esto sobrescribe la carpeta falsa)
COPY . .

# 8. Terminal abierta
CMD ["bash"]