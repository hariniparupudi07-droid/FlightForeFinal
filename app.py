import os
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, send_file
from sklearn.linear_model import LinearRegression, LogisticRegression

from data_loading import load_dataset, get_data_summary
from eda import run_eda
from preprocessing import get_preprocessed_data, PREPROCESSED_CSV
from linear_regression import run_linear_regression
from regularisation import run_regularisation
from logistic_regression import run_logistic_regression
from decision_tree import run_decision_tree
from random_forest import run_random_forest
from kmeans import run_kmeans

app = Flask(__name__)
app.jinja_env.globals.update(zip=zip)

FEATURES, X_TR, X_TE, YR_TR, YR_TE, YC_TR, YC_TE, DF_CLEAN, SCALER = (
    None, None, None, None, None, None, None, None, None
)

PROD_REG_MODEL = None
PROD_CLF_MODEL = None


def init_app_state():
    global FEATURES, X_TR, X_TE, YR_TR, YR_TE
    global YC_TR, YC_TE, DF_CLEAN, SCALER
    global PROD_REG_MODEL, PROD_CLF_MODEL

    try:
        (
            FEATURES,
            X_TR,
            X_TE,
            YR_TR,
            YR_TE,
            YC_TR,
            YC_TE,
            DF_CLEAN,
            SCALER
        ) = get_preprocessed_data()

        PROD_REG_MODEL = LinearRegression()
        PROD_REG_MODEL.fit(X_TR, YR_TR)

        PROD_CLF_MODEL = LogisticRegression(
            max_iter=2000,
            random_state=42
        )
        PROD_CLF_MODEL.fit(X_TR, YC_TR)

        print(
            f"Data initialized. Samples: {len(DF_CLEAN)}, "
            f"Features: {FEATURES}"
        )

    except Exception as e:
        print("Initialization note:", e)


init_app_state()


@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None

    form_data = {
        'dep_delay': 15,
        'distance': 750,
        'sched_time': 135,
        'sched_dep': 1200,
        'month': 6,
        'day': 15,
        'day_of_week': 4
    }

    if request.method == 'POST' or request.args.get('predict'):
        try:
            dep_delay = float(request.values.get('dep_delay', 15))
            distance = float(request.values.get('distance', 750))
            sched_time = float(request.values.get('sched_time', 135))
            sched_dep = float(request.values.get('sched_dep', 1200))
            month = int(request.values.get('month', 6))
            day = int(request.values.get('day', 15))
            day_of_week = int(request.values.get('day_of_week', 4))

            form_data = {
                'dep_delay': dep_delay,
                'distance': distance,
                'sched_time': sched_time,
                'sched_dep': sched_dep,
                'month': month,
                'day': day,
                'day_of_week': day_of_week
            }

            raw_input = pd.DataFrame([{
                'MONTH': month,
                'DAY': day,
                'DAY_OF_WEEK': day_of_week,
                'SCHEDULED_DEPARTURE': sched_dep,
                'DEPARTURE_DELAY': dep_delay,
                'DISTANCE': distance,
                'SCHEDULED_TIME': sched_time
            }])[FEATURES]

            scaled_input = SCALER.transform(raw_input)

            pred_mins = PROD_REG_MODEL.predict(scaled_input)[0]
            prob_delayed = PROD_CLF_MODEL.predict_proba(
                scaled_input
            )[0, 1]

            prediction = {
                'expected_delay_mins': round(float(pred_mins), 1),
                'prob_delayed_pct': round(float(prob_delayed * 100), 1),
                'is_delayed': bool(
                    prob_delayed >= 0.50 or pred_mins >= 15.0
                ),
                'status_label': (
                    'Likely Delayed'
                    if prob_delayed >= 0.50 or pred_mins >= 15.0
                    else 'Likely On-Time'
                )
            }

        except Exception as e:
            prediction = {'error': str(e)}

    return render_template(
        'index.html',
        prediction=prediction,
        form_data=form_data
    )


@app.route('/data_loading')
def data_loading():
    try:
        df = load_dataset()
        summary = get_data_summary(df)
        return render_template('data_loading.html', summary=summary)
    except Exception as e:
        return render_template('data_loading.html', error=str(e))


@app.route('/eda')
def eda():
    try:
        df = load_dataset()
        result = run_eda(df)
        return render_template('eda.html', result=result)
    except Exception as e:
        return render_template('eda.html', error=str(e))


@app.route('/preprocessing')
def preprocessing():
    if DF_CLEAN is None:
        return render_template(
            'preprocessing.html',
            error="Preprocessed data is not ready."
        )

    try:
        preview_rows = DF_CLEAN.head(8).to_dict(orient='records')

        info = {
            'total_cleaned_samples': len(DF_CLEAN),
            'train_samples': len(X_TR),
            'test_samples': len(X_TE),
            'features': FEATURES,
            'scaled_features':
                "StandardScaler applied (Zero mean, unit variance)",
            'target_reg':
                "ARRIVAL_DELAY (Continuous arrival delay in minutes)",
            'target_clf':
                "IS_DELAYED (>= 15 minutes = 1, On-Time = 0)",
            'delay_ratio':
                f"{(YC_TR.sum() / len(YC_TR) * 100):.2f}% delayed",
            'preview_columns':
                [c for c in DF_CLEAN.columns
                 if not c.endswith('_SCALED')][:8],
            'preview_rows': preview_rows,
            'csv_exists': os.path.exists(PREPROCESSED_CSV),
            'csv_size_mb':
                round(
                    os.path.getsize(PREPROCESSED_CSV) /
                    (1024 * 1024),
                    2
                )
                if os.path.exists(PREPROCESSED_CSV)
                else 0
        }

        return render_template(
            'preprocessing.html',
            info=info
        )

    except Exception as e:
        return render_template(
            'preprocessing.html',
            error=str(e)
        )


@app.route('/download_preprocessed')
def download_preprocessed():
    try:
        if not os.path.exists(PREPROCESSED_CSV):
            from preprocessing import generate_and_save_preprocessed_csv
            generate_and_save_preprocessed_csv()

        return send_file(
            PREPROCESSED_CSV,
            as_attachment=True,
            download_name='preprocessed_flights.csv',
            mimetype='text/csv'
        )

    except Exception as e:
        return f"Error downloading file: {e}", 500


@app.route('/linear_regression')
def linear_regression():
    if X_TR is None:
        return render_template(
            'linear_regression.html',
            error="Model data unavailable."
        )

    selected_feature = request.args.get('feature', 'ALL')

    metrics = run_linear_regression(
        X_TR,
        X_TE,
        YR_TR,
        YR_TE,
        feature_names=FEATURES,
        selected_feature=selected_feature,
        df_clean=DF_CLEAN
    )

    return render_template(
        'linear_regression.html',
        metrics=metrics,
        features=FEATURES,
        selected_feature=selected_feature
    )


@app.route('/regularisation')
def regularisation():
    if X_TR is None:
        return render_template(
            'regularisation.html',
            error="Model data unavailable."
        )

    alpha = request.args.get('alpha', '1.0')
    c_value = request.args.get('c', '1.0')

    try:
        output = run_regularisation(
            X_train_reg=X_TR,
            X_test_reg=X_TE,
            y_train_reg=YR_TR,
            y_test_reg=YR_TE,

            X_train_clf=X_TR,
            X_test_clf=X_TE,
            y_train_clf=YC_TR,
            y_test_clf=YC_TE,

            alpha=alpha,
            C=c_value,
            feature_names=FEATURES
        )

        return render_template(
            'regularisation.html',
            data=output,
            current_alpha=output[
                'linear_regression'
            ]['alpha'],
            current_c=output[
                'logistic_regression'
            ]['C']
        )

    except Exception as e:
        return render_template(
            'regularisation.html',
            error=str(e)
        )


@app.route('/logistic_regression')
def logistic_regression():
    if X_TR is None:
        return render_template(
            'logistic_regression.html',
            error="Model data unavailable."
        )

    metrics = run_logistic_regression(
        X_TR,
        X_TE,
        YC_TR,
        YC_TE,
        feature_names=FEATURES
    )

    return render_template(
        'logistic_regression.html',
        metrics=metrics
    )


@app.route('/decision_tree')
def decision_tree():
    if X_TR is None:
        return render_template(
            'decision_tree.html',
            error="Model data unavailable."
        )

    max_depth = int(
        request.args.get('max_depth', 6)
    )

    metrics = run_decision_tree(
        X_TR,
        X_TE,
        YC_TR,
        YC_TE,
        max_depth=max_depth
    )

    return render_template(
        'decision_tree.html',
        metrics=metrics,
        max_depth=max_depth
    )


@app.route('/random_forest')
def random_forest():
    if X_TR is None:
        return render_template(
            'random_forest.html',
            error="Model data unavailable."
        )

    n_estimators = int(
        request.args.get('n_estimators', 50)
    )

    metrics = run_random_forest(
        X_TR,
        X_TE,
        YC_TR,
        YC_TE,
        n_estimators=n_estimators
    )

    return render_template(
        'random_forest.html',
        metrics=metrics,
        n_estimators=n_estimators
    )


@app.route('/kmeans', methods=['GET', 'POST'])
def kmeans_view():
    method = request.values.get(
        'method',
        'silhouette'
    )
    manual_k = request.values.get('manual_k')

    try:
        result = run_kmeans(
            method=method,
            manual_k=manual_k
        )

        return render_template(
            'kmeans.html',
            result=result,
            method=method
        )

    except Exception as e:
        return render_template(
            'kmeans.html',
            error=str(e),
            method=method
        )


if __name__ == '__main__':
    app.run(
        debug=True,
        port=5000
    )
