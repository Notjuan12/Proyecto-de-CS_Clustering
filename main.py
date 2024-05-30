import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer as tfid
from sklearn.cluster import KMeans as km
from sklearn.decomposition import PCA
import Datos_cluster



def inicializar_csv():  # usar solo los datos de moves para el tdf
    datos = pd.read_csv("smogon - smogonyaseparado.csv")
    datos_moves = datos.drop(["Pokemon", "url", "texto"], axis=1)
    nombres = datos.drop(["moves", "url", "texto"], axis=1)
    return datos_moves, nombres


def inicializar_vectfid(datos):  #con stopwords no agrupa bien
    vectdf = tfid(ngram_range=(2, 3))
    vectdfesultado = vectdf.fit_transform(datos["moves"])
    vocabulario = vectdf.vocabulary_
    tokens = len(vectdf.vocabulary_)
    return vectdfesultado, vocabulario, tokens


def tabla_de_frecuencias(moves):
    tfidresult, vocabulario, tokens = inicializar_vectfid(moves)
    matriz = tfidresult.toarray()
    tabla_frecuencias = pd.DataFrame(data=matriz, columns=sorted(vocabulario))
    return tabla_frecuencias, matriz, tokens


def pca_preg2(tabla, km_colum, pokemones):
    pca = PCA(18)
    x_pca = pca.fit_transform(tabla)
    cabeceras = Datos_cluster.cabeceras()
    pca_data = pd.DataFrame(data=x_pca, columns=cabeceras)
    colum_pca = inicializar_Kmeas(pca_data)
    pca_data["grupo_kmean_tfid"] = km_colum
    pca_data["grupo_kmean_pca"] = colum_pca
    pca_data["nombres"] = pokemones

    return pca_data


def inicializar_Kmeas(tablaf):
    veckm = km(n_clusters=18, n_init=20)
    tablaf = veckm.fit_predict(tablaf, sample_weight=10)
    return tablaf


def procesar_colum(moves):
    lista_tipos = Datos_cluster.tipos_pokemones()


def main():
    moves, pokemons = inicializar_csv()
    procesar_colum(moves)
    tabla_frecuencias, matriz, tokens = tabla_de_frecuencias(moves)
    kmean_colum = inicializar_Kmeas(tabla_frecuencias)
    pca_pokemons = pca_preg2(tabla_frecuencias,kmean_colum,pokemons)
    print(matriz)
    print(tabla_frecuencias)
    print("Numero de tokens: ", tokens)
    tabla_frecuencias["tipo"] = kmean_colum
    tabla_frecuencias["nombre"] = pokemons
    print(tabla_frecuencias)
    pca_pokemons.to_csv("pca_comparacion_kmean.csv")
    tabla_frecuencias.to_csv("Cluster_p_sin_nombre")



if __name__ == "__main__":
    main()
