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
RUN uv pip install --system -e .

# 7. Copiar el código real (esto sobrescribe la carpeta falsa)
COPY . .

# 8. Terminal abierta
CMD ["bash"]