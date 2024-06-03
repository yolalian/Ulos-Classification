from flask import Flask, render_template, request, redirect, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os

app = Flask(__name__)

# Load the trained model
model = load_model('trained_model.h5')
model.make_predict_function()  # Necessary for some environments

# Dictionary to map predicted class index to class label and description
dic = {
    0: {'label': 'Batak_Angkola', 'description': 'Batak Angkola merupakan salah satu kelompok etnis Batak. Tanah ulayat Batak Angkola berada di wilayah selatan Tapanuli, yakni meliputi Kabupaten Tapanuli Selatan, Kota Padang Sidempuan, Kabupaten Padang Lawas Utara, Kabupaten Padang Lawas, dan sebagian Kabupaten Mandailing Natal.'},
    1: {'label': 'Batak_Karo', 'description': 'Batak Karo adalah salah satu kelompok etnis Batak yang menyebar dan menetap di Taneh Karo. Etnis ini merupakan salah satu etnis terbesar di Sumatera Utara. Nama etnis ini juga dijadikan sebagai nama salah satu kabupaten di Provinsi Sumatera Utara, yaitu Kabupaten Karo. '},
    2: {'label': 'Batak_Mandailing', 'description': 'Batak Mandailing merupakan salah satu kelompok etnik pribumi yang menghuni daerah selatan Provinsi Sumatera Utara. Mereka pernah berada di bawah pengaruh Kaum Padri dari Minangkabau, sehingga secara kultural etnis ini dipengaruhi oleh budaya agama Islam.'},
    3: {'label': 'Batak_PakPak', 'description': 'Batak Pakpak adalah salah satu kelompok etnis Batak yang menyebar dan menetap di wilayah Dairi, Pakpak Bharat, Humbang Hasundutan, dan Tapanuli Tengah di Sumatera Utara, serta sebagian wilayah Aceh Singkil dan Subulussalam di Aceh.'},
    4: {'label': 'Batak_Simalungun', 'description': 'Batak Simalungun merupakan salah satu kelompok etnis Batak yang menyebar dan menetap di Kabupaten Simalungun dan sekitarnya di Sumatera Utara. Sepanjang sejarah, etnis Batak Simalungun terbagi ke dalam beberapa kerajaan. Marga asli penduduk Simalungun adalah Damanik, dan tiga marga pendatang yaitu Saragih, Sinaga, dan Purba. Kemudian marga-marga tersebut menjadi empat marga utama di Simalungun.'},
    5: {'label': 'Batak_Toba', 'description': 'Batak yang berasal dari Provinsi Sumatera Utara, Indonesia. Wilayah persebaran utama kelompok etnis Batak Toba meliputi Kabupaten Samosir, Kabupaten Toba, Kabupaten Humbang Hasundutan, Kabupaten Tapanuli Utara, dan Kabupaten Tapanuli Tengah.'}
}

# Route for upload page
@app.route("/", methods=['GET', 'POST'])
def upload_page():
    return render_template("upload.html")

# Route for handling image submission and classification
@app.route("/submit", methods=['POST'])
def classify_image():
    if request.method == 'POST':
        img = request.files['my_image']
        img_path = os.path.join("static", img.filename)
        img.save(img_path)
        prediction = predict_label(img_path)
        return render_template("classification.html", prediction=prediction, img_path=img_path)

# Route for deleting image
@app.route("/delete", methods=['POST'])
def delete_image():
    img_path = request.form['img_path']
    if os.path.exists(img_path):
        os.remove(img_path)
    return redirect(url_for('upload_page'))

def predict_label(img_path):
    img = load_img(img_path, target_size=(224, 224))  # Ensure the size matches the model input
    img = img_to_array(img)
    img = img / 255.0  # Normalize the image
    img = np.expand_dims(img, axis=0)  # Add batch dimension

    predictions = model.predict(img)
    predicted_class_index = np.argmax(predictions)
    return dic[predicted_class_index]

if __name__ == '__main__':
    app.run(debug=True)
