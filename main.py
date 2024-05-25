import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer as tfid
from sklearn.cluster import KMeans as km


def inicializar_csv():  # usar solo los datos de moves para el tdf
    datos = pd.read_csv("smogon - smogonyaseparado.csv")
    datos_moves = datos.drop(["Pokemon", "url", "texto"], axis=1)
    nombres = datos.drop(["moves", "url", "texto"], axis=1)
    return datos_moves, nombres


def tipos_pokemones():
    tipos = ["Normal", "Fire", "Water", "Electric", "Grass", "Ice", "Fighting", "Poison", "Ground", "Flying", "Psychic",
             "Bug", "Rock", "Ghost", "Dragon", "Dark", "Steel", "Fairy","normal", "fire", "water", "electric", "grass", "ice", "fighting", "poison", "ground", "flying", "psychic",
             "bug", "rock", "ghost", "dragon", "dark", "steel", "fairy"]

def inicializar_vectfid(datos):
    vectdf = tfid(ngram_range=(2, 3))
    vectdfesultado = vectdf.fit_transform(datos["moves"])
    vocabulario = vectdf.vocabulary_
    tokens = len(vectdf.vocabulary_)
    return vectdfesultado, vocabulario, tokens


def inicializar_Kmeas(tablaf):
    veckm = km(n_clusters=18)
    tablaf = veckm.fit_predict(tablaf, sample_weight=10)
    return tablaf


def main():
    moves, pokemons = inicializar_csv()
    tfidresult, vocabulario, tokens = inicializar_vectfid(moves)
    matriz = tfidresult.toarray()
    tabla_frecuencias = pd.DataFrame(data=matriz, columns=sorted(vocabulario))
    kmean_colum = inicializar_Kmeas(tabla_frecuencias)
    print(matriz)
    print(tabla_frecuencias)
    print("Numero de tokens: ", tokens)
    tabla_frecuencias["tipo"] = kmean_colum
    tabla_frecuencias["nombre"] = pokemons
    print(tabla_frecuencias)
    tabla_frecuencias.to_csv("Cluster_p_sin_nombre")


if __name__ == "__main__":
    main()
