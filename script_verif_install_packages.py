import importlib.util
import subprocess
import sys

def package_verification(module, pip_name): 
    """
    Vérifie si un package est installé et affiche ses informations.
    """
    if importlib.util.find_spec(module): # importlib.util.find_spec() retourne None si le module n'est pas trouvé
        # Le package est installé : afficher ses infos
        print("Le package est bien installé, voici ses informations:\n")
        subprocess.run([sys.executable, "-m", "pip", "show", pip_name]) # subprocess.run() exécute la commande pip show pour afficher les informations du package
    else:
        print(f"Le package {pip_name} n'est pas installé. Taper: pip install {pip_name} pour l'installer.")

if __name__ == "__main__":
    packages = {
    "langchain": "langchain",
    "faiss": "faiss-cpu",
    "mistralai": "mistralai",    
}
    
    for module, pip_name in packages.items():
        print(f"==============Vérification du package {pip_name}==============\n")
        package_verification(module, pip_name)
        print()

    