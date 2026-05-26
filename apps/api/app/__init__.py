# Questo file (vuoto) trasforma la cartella `app/` in un **package Python**:
# significa che `import app` e `from app.main import app` funzionano.
#
# In Python, qualunque cartella con un `__init__.py` (anche vuoto) è
# importabile. Senza, è solo una cartella ignorata dall'import system.
#
# Equivalente concettuale: il `package.json` di un sotto-package, ma molto
# più leggero — qui il marker da solo basta.
