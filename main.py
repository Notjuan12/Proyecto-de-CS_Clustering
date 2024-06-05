import pandas as pd
from sklearn.cluster import KMeans as km
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer as tfid
import Datos_cluster


def inicializar_csv():  # usar solo los datos de moves para el tdf
    datos = pd.read_csv("smogon - smogonyaseparado.csv")
    datos_moves = datos.drop(["Pokemon", "url", "texto"], axis=1)
    nombres = datos.drop(["moves", "url", "texto"], axis=1)
    return datos_moves, nombres


def inicializar_vectfid(datos):  # con stopwords no agrupa bien , prueben deafult (2,3)
    vectdf = tfid(ngram_range=(1, 3))
    vectdfesultado = vectdf.fit_transform(datos["moves"])
    vocabulario = vectdf.vocabulary_
    tokens = len(vectdf.vocabulary_)
    return vectdfesultado, vocabulario, tokens


def tabla_de_frecuencias(moves):  # recibe datos del csv y luego lo pasa a los demas funciones
    tfidresult, vocabulario, tokens = inicializar_vectfid(moves)
    matriz = tfidresult.toarray()
    tabla_frecuencias = pd.DataFrame(data=matriz, columns=sorted(vocabulario))
    return tabla_frecuencias, matriz, tokens, vocabulario


def pca_preg2(tabla, km_colum, pokemones):  # pca -> kmean
    pca = PCA(18)
    x_pca = pca.fit_transform(tabla)
    cabeceras = Datos_cluster.cabeceras()
    pca_data = pd.DataFrame(data=x_pca, columns=cabeceras)
    colum_pca = inicializar_Kmeas(pca_data)
    pca_data["grupo_kmean_tfid"] = km_colum
    pca_data["grupo_kmean_pca"] = colum_pca
    pca_data["nombres"] = pokemones

    return pca_data


def inicializar_Kmeas(tablaf):  # kmean <- recibe valores del tfid y pca
    veckm = km(n_clusters=18, n_init=20)
    tablaf = veckm.fit_predict(tablaf, sample_weight=20)
    return tablaf


def separar_tipos(data):  # separar los tipos de pokemones de la columna "moves"
    tipos = Datos_cluster.tipos_pokemones()

    def extraer_palabras(oracion, palabras):  # stack overflow 2019
        return ' '.join(palabra for palabra in oracion.split() if palabra in palabras)

    data['texto'] = data['moves'].apply(lambda oracion: extraer_palabras(oracion, tipos))
    return data


def agrupar_texto_tipos(data):
    vectdf = tfid(ngram_range=(1, 1))
    vectdfesultado = vectdf.fit_transform(data["texto"])
    vocabulario = vectdf.vocabulary_
    tokens = len(vectdf.vocabulary_)
    matriz = vectdfesultado.toarray()
    tabla_frecuencias = pd.DataFrame(data=matriz, columns=sorted(vocabulario))
    column_text = inicializar_Kmeas(tabla_frecuencias)
    tabla_frecuencias["cluster"] = column_text
    return tabla_frecuencias, tokens, vocabulario


def main():  # ejecucion de todas las funciones
    moves, pokemons = inicializar_csv()
    tabla_frecuencias, matriz, tokens, vocabulario = tabla_de_frecuencias(moves)  # vocabulario, 4 n-gramas,fijarse
    kmean_colum = inicializar_Kmeas(tabla_frecuencias)
    tipo_filtrados = separar_tipos(moves)
    pca_pokemons = pca_preg2(tabla_frecuencias, kmean_colum, pokemons)
    Datos_cluster.separador("Matriz TDIF")
    print(matriz)
    Datos_cluster.separador("Vocabulario y Tokens")
    print(tokens)
    print(vocabulario)
    Datos_cluster.separador("T.frecuencias sin Cluster")
    print(tabla_frecuencias)
    print("Numero de tokens: ", tokens)
    tabla_frecuencias["tipo"] = kmean_colum
    tabla_frecuencias["nombre"] = pokemons
    Datos_cluster.separador("T.frecuencias con Cluster")
    print(tabla_frecuencias)
    Datos_cluster.separador("PCA")
    print(pca_pokemons)
    Datos_cluster.separador(r"Texto filtrado")
    print(tipo_filtrados)
    tabla_frecuencias_text, tokens, vocabulario = agrupar_texto_tipos(tipo_filtrados)
    Datos_cluster.separador(("Texto filtrado_cluster"))
    print(vocabulario)
    print(tabla_frecuencias)
    print("numero de tokens ", tokens)
    pca_pokemons.to_csv("pca_comparacion_kmean.csv")
    tabla_frecuencias.to_csv("Cluster_p_sin_nombre.csv")
    tabla_frecuencias_text.to_csv("Tipos_filtrados_2.csv")


if __name__ == "__main__":
    main()
