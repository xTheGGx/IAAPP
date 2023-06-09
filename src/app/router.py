import os
import pandas as pd    
from flask import Blueprint, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename
from src.api.ia.metricasModule import Metricas
from src.api.ia.clusteringModule import Clustering
from src.api.ia.aprioriModule import Apriori
from src.api.ia.treesModule import execProgTree, createGrafProg
router = Blueprint('router', __name__)

@router.route('/')
@router.route('/home')
def home():
    return render_template('home.html')

@router.route('/menu')
def menu():
    algorithms = [
        {
            'title': 'Apriori',
            'description': 'Descubre patrones ocultos en grandes conjuntos de datos',
            'image': 'img/logo.png',
            'link' : 'apriori'

        },
        {
            'title': 'Métricas de distancia',
            'description': 'Observa cómo revelan similitudes y diferencias entre objetos en el análisis de datos',
            'image': 'img/logo.png',
            'link' : 'metricas'
        },
        {
            'title': 'Clustering',
            'description': 'Segmenta tus datos y toma decisiones más informadas',
            'image': 'img/logo.png',
            'link': 'clustering'
        },
        {
            'title': 'Árboles de desición',
            'description': 'Explora la magia de los árboles y bosques aleatorios para predecir.',
            'image': 'img/logo.png',
            'link': 'tree'
        },
        {
            'title': 'Bosques aleatorios',
            'description': 'Explora la magia de los árboles y bosques aleatorios para predecir.',
            'image': 'img/logo.png',
            'link': 'forest'
        }
    ]
    return render_template('menu.html', algorithms=algorithms)

@router.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        file = request.files['file']
        if file and file.filename.endswith('.csv'):
            filename = secure_filename(file.filename)
            file.save(os.path.join('input', filename))
            return render_template('upload.html', success_message='Archivo subido correctamente.')
        else:
            error_message = 'Error: El archivo debe tener extensión .csv.'
            return render_template('upload.html', error_message=error_message)
    return render_template('upload.html')

@router.route('/apriori')
def aprioriAlg():
    input_folder = 'input'
    files = os.listdir(input_folder)
    csv_files = [file for file in files if file.endswith('.csv')]
    return render_template('apriori.html', csv_files=csv_files)

@router.route('/apriori/process', methods=['POST'])
def aprioriProcess():
    file = request.form['file']
    
    # Obtener los valores ingresados por el usuario
    support = float(request.form['support'])
    confidence = float(request.form['confidence'])
    lift = float(request.form['lift'])

    aprioriModule = Apriori(fileName=file)
    # Debería ser ejecutado desde un endpoint, Evita ejecutarlo en las rutas de la aplicación ya que ocasiona un cuello de botella antes de ser pintado el template, ya que tiene que esperar a Lista_html además de la imagen del plot
    Lista_html = aprioriModule.createFrecuencyTable().to_html()
    # Procura solo enviar parámetros que no necesitan un preproceso, un ejemplo son las configuraciones del algoritmo como soporte, confianza o elevación
    return render_template('aprioriProcess.html',
                           dataframe=Lista_html,
                           file= file, support=support, confidence=confidence, lift=lift)

@router.route('/metricas')
def metricas():
    input_folder = 'input'
    files = os.listdir(input_folder)
    csv_files = [file for file in files if file.endswith('.csv')]
    return render_template('metricas.html', csv_files=csv_files)



@router.route('/metricas/process', methods=['POST'])
def metricasProcess():
    selected_file = request.form['file']
    option = request.form['option']
    lim_inf = int(request.form['limInf'])
    lim_sup = int(request.form['limSup'])
    dist_a = int(request.form['distA'])
    dist_b = int(request.form['distB'])
    print(option)
    if option == "euclidean":
        return redirect(url_for('router.metricasEuclidiana', selected_file=selected_file, lim_inf=lim_inf, lim_sup=lim_sup, dist_a=dist_a, dist_b=dist_b, option = option))
    elif option == "chebyshev":
        return redirect(url_for('router.metricasChebyshev', selected_file=selected_file, lim_inf=lim_inf, lim_sup=lim_sup, dist_a=dist_a, dist_b=dist_b, option = option))
    elif option == "manhattan":
        return redirect(url_for('router.metricasManhattan', selected_file=selected_file, lim_inf=lim_inf, lim_sup=lim_sup, dist_a=dist_a, dist_b=dist_b, option = option))
    elif option == "minkowski":
        return redirect(url_for('router.metricasMinkowsky', selected_file=selected_file, lim_inf=lim_inf, lim_sup=lim_sup, dist_a=dist_a, dist_b=dist_b, option = option))



#Euclidiana, Chebyshev, Manhattan, Minkowski

@router.route("/metricas/euclidean", methods=['GET', 'POST'])
def metricasEuclidiana():
    option = request.args.get('option')
    selected_file = request.args.get('selected_file')
    distA = int(request.args.get('dist_a'))
    distB = int(request.args.get('dist_b'))
    limInferior = int(request.args.get('lim_inf'))
    limSuperior = int(request.args.get('lim_sup'))
    metricas = Metricas(selected_file)
    MEstandarizada = metricas.createME()
    MEuclidiana = metricas.createM(MEstandarizada,option)
    MParcial = metricas.partialDistance(MEuclidiana, limInferior,limSuperior,option)
    Dst = metricas.distEuclidean(MEuclidiana, distA, distB)

    # Cambiar la ejecución del algoritmo hacia un endpoint que sea pedido desde el template, puede observar el ejemplo de apriori
    return render_template("metricasEuclidiana.html"
                           ,euclidianSample=MEstandarizada.to_html()
                           ,MEuclidiana=MEuclidiana.to_html()
                           , MParcial = MParcial.to_html()
                           , Dst = Dst, A = distA, B =distB)


#Euclidiana, Chebyshev, Manhattan, Minkowski


@router.route("/metricas/chebyshev")
def metricasChebyshev():
    option = request.args.get('option')
    selected_file = request.args.get('selected_file')
    distA = int(request.args.get('dist_a'))
    distB = int(request.args.get('dist_b'))
    limInferior = int(request.args.get('lim_inf'))
    limSuperior = int(request.args.get('lim_sup'))
    metricas = Metricas(selected_file)

    MEstandarizada = metricas.createME()
    MEuclidiana = metricas.createM(MEstandarizada,option)
    MParcial = metricas.partialDistance(MEuclidiana, limInferior,limSuperior,option)
    Dst = metricas.distChebyshev(MEuclidiana, distA, distB)


    return render_template("metricasEuclidiana.html"
                           ,euclidianSample=MEstandarizada.to_html()
                           ,MEuclidiana=MEuclidiana.to_html()
                           , MParcial = MParcial.to_html()
                           , Dst = Dst, A = distA, B =distB)

@router.route("/metricas/manhattan")
def metricasManhattan():
    option = 'cityblock'
    selected_file = request.args.get('selected_file')
    distA = int(request.args.get('dist_a'))
    distB = int(request.args.get('dist_b'))
    limInferior = int(request.args.get('lim_inf'))
    limSuperior = int(request.args.get('lim_sup'))
    metricas = Metricas(selected_file)

    MEstandarizada = metricas.createME()
    MEuclidiana = metricas.createM(MEstandarizada,option)
    MParcial = metricas.partialDistance(MEuclidiana, limInferior,limSuperior,option)
    Dst = metricas.distManhattan(MEuclidiana, distA, distB)


    return render_template("metricasEuclidiana.html"
                           ,euclidianSample=MEstandarizada.to_html()
                           ,MEuclidiana=MEuclidiana.to_html()
                           , MParcial = MParcial.to_html()
                           , Dst = Dst, A = distA, B =distB)

@router.route("/metricas/minkowsky")
def metricasMinkowsky():
    option = request.args.get('option')
    selected_file = request.args.get('selected_file')
    distA = int(request.args.get('dist_a'))
    distB = int(request.args.get('dist_b'))
    limInferior = int(request.args.get('lim_inf'))
    limSuperior = int(request.args.get('lim_sup'))
    metricas = Metricas(selected_file)

    MEstandarizada = metricas.createME()
    MEuclidiana = metricas.createM(MEstandarizada,option)
    MParcial = metricas.partialDistance(MEuclidiana, limInferior,limSuperior,option)
    Dst = metricas.distMinkowski(MEuclidiana, distA, distB)


    return render_template("metricasEuclidiana.html"
                           ,euclidianSample=MEstandarizada.to_html()
                           ,MEuclidiana=MEuclidiana.to_html()
                           , MParcial = MParcial.to_html()
                           , Dst = Dst, A = distA, B =distB)

@router.route('/clustering')
def clustering():
    input_folder = 'input'
    files = os.listdir(input_folder)
    csv_files = [file for file in files if file.endswith('.csv')]
    return render_template('clustering.html', csv_files=csv_files)


@router.route('/clustering/process', methods=['GET', 'POST'])
def clusteringProcess():
    selected_file = request.form['file']
    option = request.form['option']
    
    if option == "jerarquico":
        return redirect(url_for('router.clusteringJerarquico', selected_file=selected_file, option = option))
    elif option == "particional":
        return redirect(url_for('router.clusteringParticional', selected_file=selected_file, option = option))


@router.route('/clustering/jerarquico')
def clusteringJerarquico():
    option = request.args.get('option')
    selected_file = request.args.get('selected_file')

    cluster = Clustering(selected_file)
    corrMatrix = cluster.createCorrMatrix()
    hMap = cluster.createHeatMap(corrMatrix)
    matrizVar = cluster.createMVar()
    Mestandar = cluster.MEstandarizada(matrizVar)
    hCluster = cluster.hierarchyClusterPNG(Mestandar)
    matrix_html = cluster.hierarchyClusterArray(Mestandar)

    return render_template("jerarquico.html",
                        hMap=hMap,
                        corrMatrix=corrMatrix,
                        matrizVar=matrizVar,
                        matrix_html=matrix_html,
                        hCluster=hCluster)



@router.route('/clustering/particional', methods=['GET', 'POST'])
def clusteringParticional():
    option = request.args.get('option')
    selected_file = request.args.get('selected_file')
    cluster = Clustering(selected_file)
    vars = []
    corrMatrix = cluster.createCorrMatrix(cluster.delNoNums())
    var = cluster.getColumns()
    hMap = cluster.createHeatMap(corrMatrix)
    if request.method == 'POST':
        vars = request.form.getlist('vari')
        metric = request.form['metric']
        mVar = cluster.createMVar(vars)
        mEst = cluster.MEstandarizada(mVar)
        data = cluster.df
        test = cluster.hierarchyClusterArray(mEst,metric)
        data['clusterH'] = test.labels_
        #Crea img de cluster
        pRoute = cluster. PartitionalClusterPNG(mEst,metric)
        centroidsP = data.groupby(['clusterH'])[var].mean()
        centroidsP = centroidsP.to_html()
        
        return render_template('particional_results.html'
                               ,pRoute = pRoute
                               ,centroidsP =  centroidsP
                               , variables = vars
                               , metric = metric
                               )
    return render_template("particional.html", hMap = hMap,
                           variables=var)

@router.route('/clustering/particional/results', methods=['GET', 'POST'])
def clusteringParticionalResults():
    h = 2



@router.route('/tree')
def tree():
    input_folder = 'input'
    files = os.listdir(input_folder)
    csv_files = [file for file in files if file.endswith('.csv')]
    return render_template('tree.html', csv_files=csv_files)

@router.route('/tree/process', methods=['GET', 'POST'])
def treeProcess():
    selected_file = request.form['file']
    option = request.form['option']
    
    if option == "clasificacion":
        return redirect(url_for('router.clusteringJerarquico', selected_file=selected_file, option = option))
    elif option == "pronostico":
        return redirect(url_for('router.pronosTree', selected_file=selected_file, option = option))

@router.route ('/tree/prognosis', methods=['GET', 'POST'])
def pronosTree():

    createGrafProg()
    treerep = None
    treescore = None
    predic = None

    if request.method == 'POST':
        depth = int(request.form['parametro1'])
        split = int(request.form['parametro2'])
        leaf = int(request.form['parametro3'])
        open = float(request.form['parametro4'])
        high = float(request.form['parametro5'])
        low = float(request.form['parametro6'])

        arbrep, arbscore, predic = execProgTree(depth,split,leaf,open,high,low)

    return render_template('prognosis.html', arbrep=treerep, arbscore=treescore, predic=predic)
