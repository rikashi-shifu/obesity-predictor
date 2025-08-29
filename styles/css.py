def get_custom_css():
    return """
    <style>
        .main-header {
            font-size: 3rem;
            color: #1f77b4;
            text-align: center;
            margin-bottom: 2rem;
            font-weight: bold;
        }
        .sub-header {
            font-size: 1.5rem;
            color: #d35400;
            margin-bottom: 1rem;
            font-weight: bold;
        }
        .info-box {
            background-color: #f8f9fa;
            color: #2c3e50;
            padding: 1.5rem;
            border-radius: 0.7rem;
            border-left: 5px solid #3498db;
            margin: 1rem 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .success-box {
            background-color: #e8f5e8;
            color: #1b5e20;
            padding: 1.5rem;
            border-radius: 0.7rem;
            border-left: 5px solid #4caf50;
            margin: 1rem 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .warning-box {
            background-color: #fff8e1;
            color: #e65100;
            padding: 1.5rem;
            border-radius: 0.7rem;
            border-left: 5px solid #ff9800;
            margin: 1rem 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .prediction-result {
            font-size: 2rem;
            font-weight: bold;
            text-align: center;
            padding: 2rem;
            border-radius: 1rem;
            margin: 2rem 0;
            box-shadow: 0 4px 8px rgba(0,0,0,0.15);
        }
    </style>
    """
