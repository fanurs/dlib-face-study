# Dlib Face Study

1. Install required packages (see the imported libraries in [`app.py`](app.py)).
1. Add images of interested to [`images`](images/) folder. You should follow the structure like below:
    ```
    dlib-face-study/
        |-- images/
            |-- Alan-Turing/
                |-- "1.jpg"
            |-- Boyoung-Park/
                |-- "I Park Boyoung profile photo.png"
                |-- "640px-190619_tvN_'어비스'_종방연_박보영_(2).webp"
            |-- Chandra/
                |-- "Chandra_1.jpg"
                |-- "chandra_5.jpg"
                |-- "Sreenshot Android.png"
    ```
    Each subdirectory reprensents one person. In each subdirectory, there should be at least one image from the person. The names of the images are not important as we will be using [`glob`](https://docs.python.org/3/library/glob.html) to loop through all the images. However, the names of the subdirectories are important as we will be using them to identify the person.
1. You are now ready to run the app:
    ```console
    user@computer $ python app.py
    images\Zendaya\Zendaya_-_2019_by_Glenn_F
    Dash is running on http://127.0.0.1:5500/

    * Serving Flask app 'app' (lazy loading)
    * Environment: production
    WARNING: This is a development server. Do not use it in a production deployment.
    Use a production WSGI server instead.
    * Debug mode: off
    * Running on http://127.0.0.1:5500/ (Press CTRL+C to quit)
    127.0.0.1 - - [19/Mar/2022 22:21:06] "GET / HTTP/1.1" 200 -
    ...
    ```
    You can see the result in the browser at [`http://localhost:5500/`](http://localhost:5500/).