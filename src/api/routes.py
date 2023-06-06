import json
from flask import request, Blueprint, jsonify
from src.api.ia.aprioriModule import Apriori
from src.api.ia.metricasModule import Metricas

api = Blueprint('api', __name__)

@api.route('/apriori/data', methods=['POST'])
def aprioriData():
    file = request.json['file']
    support = float(request.json['support'])
    confidence = float(request.json['confidence'])
    lift = float(request.json['lift'])
    page = int(request.args.get('page', 1))  # Get the page number from the request
    per_page = int(request.args.get('per_page', 10))  # Get the number of items per page

    aprioriModule = Apriori(fileName=file)
    data = aprioriModule.apriori(support=support, confidence=confidence, lift=lift)
    # Calculate the start and end indices for the current page
    page_size = per_page
    start_index = (page - 1) * page_size
    end_index = start_index + page_size
    # Get the data for the current page
    paginated_data = data.iloc[start_index:end_index].to_dict(orient='records')
    json_data = json.dumps(paginated_data)
    
    return jsonify(json_data)