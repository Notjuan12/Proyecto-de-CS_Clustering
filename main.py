import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer as tfid
from sklearn.cluster import KMeans as km
from sklearn.decomposition import PCA
import Datos_cluster



def inicializar_csv():  # usar solo los datos de moves para el tdf
    datos = pd.read_csv("smogon - smogon.csv") #dasdasd
    datos_moves = datos.drop(["Pokemon", "url", "texto"], axis=1)
    nombres = datos.drop(["moves", "url", "texto"], axis=1)
    return datos_moves, nombres


def inicializar_vectfid(datos):  #con stopwords no agrupa bien
    vectdf = tfid(stop_words=(Datos_cluster.stopword_eng()),ngram_range=(1, 3))
    vectdfesultado = vectdf.fit_transform(datos["moves"])
    vocabulario = vectdf.vocabulary_
    tokens = len(vectdf.vocabulary_)
    return vectdfesultado, vocabulario, tokens


def tabla_de_frecuencias(moves):
    tfidresult, vocabulario, tokens = inicializar_vectfid(moves)
    matriz = tfidresult.toarray()
    tabla_frecuencias = pd.DataFrame(data=matriz, columns=sorted(vocabulario))
    return tabla_frecuencias, matriz, tokens, vocabulario


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
    #1.1
    moves, pokemons = inicializar_csv() #lee y separa los nombres y moves
    procesar_colum(moves) #saca del Datos_clusters los tipos
    tabla_frecuencias, matriz, tokens, listadetokens = tabla_de_frecuencias(moves) #crea tfidf , la tabla, la matriz y los tokens
    kmean_colum = inicializar_Kmeas(tabla_frecuencias) #hacer kmeans (clusters)
    print("Numero de tokens: ", tokens)
    print("Vocabulario/lista de tokens: ")
    print(listadetokens)
    print("DataFrame con la matriz tf-idf ")
    print(tabla_frecuencias)
    #csv que pide:
    tabla_frecuencias["cluster"] = kmean_colum
    tabla_frecuencias.to_csv("matriz tfidf y el cluster.csv")
    #para el analisis
    Analisis11 = pd.DataFrame()
    Analisis11["pokemons"] = pokemons
    Analisis11["clusters"] = kmean_colum
    Analisis11.to_csv("anal.csv")
    #1.2
    tabla_frecuencias.drop(["cluster"], axis=1, inplace=True)
    pca_pokemons = pca_preg2(tabla_frecuencias,kmean_colum,pokemons) #hacer pca, hacerle kmeans al pca
    #NOSE SI SIRVE TODAVIA tabla_frecuencias["tipo"] = kmean_colum

    print("sorted tab")
    #print(tabla_frecuencias.sort_values("tipo"))
    tabla_frecuencias["nombre"] = pokemons
    print(tabla_frecuencias)
    pca_pokemons.to_csv("pca_comparacion_kmean.csv")
    tabla_frecuencias.to_csv("Cluster_p_sin_nombre")



if __name__ == "__main__":
    main()
