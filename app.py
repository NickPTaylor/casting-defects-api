import pathlib
import tempfile

from flask import Flask
from flask_restful import Resource, Api, reqparse
from werkzeug.datastructures import FileStorage

from predict import predict

app = Flask(__name__)
api = Api(app)

ALLOWED_EXTENSIONS = ['.jpeg']
#IMG_FILE_NAME = 'uploads/img_data.temp.jpeg'
#tempfile.gettempdir()
class ClassifyImage(Resource):

    def post(self):
        """
       Get classification of image.
       """
        # Parse arguments and obtain file name.
        parse = reqparse.RequestParser()
        parse.add_argument('file', type=FileStorage, location='files')
        args = parse.parse_args()

        # Upload images.
        fs_object = args['file']
        if pathlib.Path(fs_object.filename).suffix not in ALLOWED_EXTENSIONS:
            return f"File extension must be: {ALLOWED_EXTENSIONS}"

        # Update file and get predictions
        with tempfile.NamedTemporaryFile() as temp_img_file:
            fs_object.save(temp_img_file.name)
            predictions = predict.get_pred_from_file(temp_img_file.name)

        return predictions


api.add_resource(ClassifyImage, '/classify/')

if __name__ == '__main__':
    app.run()
