import duckdb as db
import pandas as pd

con = duckdb.connect(database=':memory:', read_only=False)

con.execute("""
CREATE TABLE funcionarios (
    id INTEGER,
    nome VARCHAR,
    departamento VARCHAR,
    salario FLOAT
""")

produtos_df = pd.DataFrame({
    'id': [1, 2, 3],
    'nome': ['Produto A', 'Produto B', 'Produto C'],
    'preco': [10.0, 20.0, 30.0]
})

con.register('produtos_view', produtos_df)
con.execute("""INSERT INTO funcionarios VALUES
    (1, 'Alice', 'TI', 7000.0),
    (2, 'Bob', 'RH', 5000.0),
    (3, 'Charlie', 'Financeiro', 6000.0)
""")