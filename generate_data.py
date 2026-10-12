"""
Utility script to generate sample crop recommendation dataset
with realistic agronomic ranges for 22 crops (100 samples per crop = 2,200 rows).
"""

import csv
import random
import os

random.seed(42)

CROPS_SPEC = {
    'rice': {
        'N': (80, 8, 60, 100),
        'P': (48, 6, 35, 60),
        'K': (40, 3, 35, 45),
        'temperature': (23.7, 1.5, 20.0, 27.0),
        'humidity': (82.3, 1.2, 80.0, 85.0),
        'ph': (6.4, 0.4, 5.0, 7.8),
        'rainfall': (236.0, 25.0, 180.0, 300.0)
    },
    'maize': {
        'N': (78, 9, 60, 100),
        'P': (48, 6, 35, 60),
        'K': (20, 2.5, 15, 25),
        'temperature': (22.4, 2.5, 18.0, 29.0),
        'humidity': (65.1, 4.5, 55.0, 75.0),
        'ph': (6.2, 0.35, 5.5, 7.0),
        'rainfall': (85.0, 12.0, 60.0, 110.0)
    },
    'chickpea': {
        'N': (40, 8, 20, 60),
        'P': (68, 5, 55, 80),
        'K': (80, 2.5, 75, 85),
        'temperature': (18.9, 0.9, 17.0, 21.0),
        'humidity': (16.9, 1.0, 14.0, 20.0),
        'ph': (7.3, 0.5, 6.0, 8.5),
        'rainfall': (80.0, 6.5, 65.0, 95.0)
    },
    'kidneybeans': {
        'N': (21, 5, 10, 40),
        'P': (67, 6, 55, 80),
        'K': (20, 2.5, 15, 25),
        'temperature': (20.1, 2.0, 15.0, 25.0),
        'humidity': (21.6, 1.8, 18.0, 25.0),
        'ph': (5.7, 0.15, 5.5, 6.0),
        'rainfall': (106.0, 20.0, 60.0, 150.0)
    },
    'pigeonpeas': {
        'N': (21, 5, 10, 40),
        'P': (68, 5, 55, 80),
        'K': (20, 2.5, 15, 25),
        'temperature': (27.7, 3.5, 20.0, 38.0),
        'humidity': (48.1, 5.5, 30.0, 70.0),
        'ph': (5.8, 0.55, 4.5, 7.5),
        'rainfall': (149.0, 25.0, 90.0, 200.0)
    },
    'mothbeans': {
        'N': (21, 5, 10, 40),
        'P': (48, 6, 35, 60),
        'K': (20, 2.5, 15, 25),
        'temperature': (28.2, 1.8, 24.0, 32.0),
        'humidity': (53.2, 5.5, 40.0, 65.0),
        'ph': (6.8, 1.1, 3.5, 9.9),
        'rainfall': (51.0, 10.0, 30.0, 75.0)
    },
    'mungbean': {
        'N': (21, 5, 10, 40),
        'P': (48, 6, 35, 60),
        'K': (20, 2.5, 15, 25),
        'temperature': (28.5, 0.8, 27.0, 30.0),
        'humidity': (85.5, 2.5, 80.0, 90.0),
        'ph': (6.7, 0.25, 6.2, 7.2),
        'rainfall': (48.0, 5.5, 35.0, 60.0)
    },
    'blackgram': {
        'N': (40, 8, 20, 60),
        'P': (67, 6, 55, 80),
        'K': (19, 2.5, 15, 25),
        'temperature': (29.9, 2.2, 25.0, 35.0),
        'humidity': (65.1, 2.4, 60.0, 70.0),
        'ph': (7.1, 0.25, 6.5, 7.5),
        'rainfall': (68.0, 3.5, 60.0, 75.0)
    },
    'lentil': {
        'N': (19, 5, 10, 40),
        'P': (68, 6, 55, 80),
        'K': (19, 2.5, 15, 25),
        'temperature': (24.5, 2.8, 18.0, 30.0),
        'humidity': (64.8, 2.5, 60.0, 70.0),
        'ph': (6.9, 0.3, 6.5, 7.8),
        'rainfall': (45.0, 4.5, 35.0, 55.0)
    },
    'pomegranate': {
        'N': (19, 5, 10, 40),
        'P': (19, 4, 10, 30),
        'K': (40, 2.5, 35, 45),
        'temperature': (21.8, 1.8, 18.0, 25.0),
        'humidity': (90.1, 2.5, 85.0, 95.0),
        'ph': (6.4, 0.4, 5.5, 7.2),
        'rainfall': (107.0, 3.5, 100.0, 115.0)
    },
    'banana': {
        'N': (100, 9, 80, 120),
        'P': (82, 6, 70, 95),
        'K': (50, 2.5, 45, 55),
        'temperature': (27.4, 1.1, 25.0, 30.0),
        'humidity': (80.4, 2.5, 75.0, 85.0),
        'ph': (5.9, 0.25, 5.5, 6.5),
        'rainfall': (104.0, 6.5, 90.0, 120.0)
    },
    'mango': {
        'N': (20, 5, 10, 40),
        'P': (27, 5, 15, 40),
        'K': (30, 2.5, 25, 35),
        'temperature': (31.2, 2.0, 27.0, 36.0),
        'humidity': (50.2, 2.5, 45.0, 55.0),
        'ph': (5.7, 0.5, 4.5, 7.0),
        'rainfall': (94.0, 5.0, 85.0, 105.0)
    },
    'grapes': {
        'N': (23, 6, 10, 40),
        'P': (132, 6, 120, 145),
        'K': (200, 2.5, 195, 205),
        'temperature': (23.8, 6.0, 10.0, 42.0),
        'humidity': (81.9, 1.2, 80.0, 85.0),
        'ph': (6.0, 0.25, 5.5, 6.5),
        'rainfall': (69.0, 3.5, 60.0, 75.0)
    },
    'watermelon': {
        'N': (99, 9, 80, 120),
        'P': (17, 4, 10, 30),
        'K': (50, 2.5, 45, 55),
        'temperature': (25.5, 0.8, 24.0, 27.0),
        'humidity': (85.1, 2.5, 80.0, 90.0),
        'ph': (6.5, 0.25, 6.0, 7.0),
        'rainfall': (50.0, 5.0, 40.0, 60.0)
    },
    'muskmelon': {
        'N': (100, 9, 80, 120),
        'P': (17, 4, 10, 30),
        'K': (50, 2.5, 45, 55),
        'temperature': (28.6, 0.8, 27.0, 30.0),
        'humidity': (92.3, 1.2, 90.0, 95.0),
        'ph': (6.3, 0.18, 6.0, 6.8),
        'rainfall': (24.0, 2.5, 20.0, 30.0)
    },
    'apple': {
        'N': (21, 5, 10, 40),
        'P': (134, 6, 120, 145),
        'K': (200, 2.5, 195, 205),
        'temperature': (22.6, 0.8, 21.0, 24.0),
        'humidity': (92.3, 1.2, 90.0, 95.0),
        'ph': (5.9, 0.25, 5.5, 6.5),
        'rainfall': (112.0, 5.5, 100.0, 125.0)
    },
    'orange': {
        'N': (19, 5, 10, 40),
        'P': (16, 4, 10, 30),
        'K': (10, 2.5, 5, 15),
        'temperature': (22.7, 4.0, 15.0, 35.0),
        'humidity': (92.1, 1.2, 90.0, 95.0),
        'ph': (7.0, 0.5, 6.0, 8.0),
        'rainfall': (110.0, 5.0, 100.0, 120.0)
    },
    'papaya': {
        'N': (49, 9, 30, 70),
        'P': (59, 6, 45, 75),
        'K': (50, 2.5, 45, 55),
        'temperature': (33.7, 3.2, 25.0, 44.0),
        'humidity': (92.4, 1.2, 90.0, 95.0),
        'ph': (6.7, 0.15, 6.5, 7.0),
        'rainfall': (142.0, 25.0, 90.0, 250.0)
    },
    'coconut': {
        'N': (22, 5, 10, 40),
        'P': (17, 4, 10, 30),
        'K': (30, 2.5, 25, 35),
        'temperature': (27.4, 0.9, 25.0, 29.0),
        'humidity': (97.2, 1.2, 95.0, 100.0),
        'ph': (5.9, 0.25, 5.5, 6.5),
        'rainfall': (175.0, 22.0, 130.0, 225.0)
    },
    'cotton': {
        'N': (118, 9, 100, 140),
        'P': (46, 6, 35, 60),
        'K': (19, 2.5, 15, 25),
        'temperature': (23.9, 1.0, 22.0, 26.0),
        'humidity': (79.8, 2.5, 75.0, 85.0),
        'ph': (6.9, 0.5, 6.0, 8.0),
        'rainfall': (80.0, 9.0, 60.0, 100.0)
    },
    'jute': {
        'N': (78, 9, 60, 100),
        'P': (46, 6, 35, 60),
        'K': (40, 2.5, 35, 45),
        'temperature': (24.9, 1.0, 23.0, 27.0),
        'humidity': (79.6, 4.5, 70.0, 90.0),
        'ph': (6.7, 0.35, 6.0, 7.5),
        'rainfall': (174.0, 11.0, 150.0, 200.0)
    },
    'coffee': {
        'N': (101, 9, 80, 120),
        'P': (28, 6, 15, 40),
        'K': (30, 2.5, 25, 35),
        'temperature': (25.5, 1.2, 23.0, 28.0),
        'humidity': (58.8, 4.8, 50.0, 70.0),
        'ph': (6.8, 0.35, 6.0, 7.5),
        'rainfall': (158.0, 20.0, 115.0, 200.0)
    }
}

def sample_value(mean, std, min_val, max_val, is_int=False):
    val = random.gauss(mean, std)
    val = max(min_val, min(max_val, val))
    if is_int:
        return int(round(val))
    return round(val, 2)

def generate_csv(output_path, samples_per_crop=100):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    rows = []
    headers = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall', 'label']

    for crop, params in CROPS_SPEC.items():
        for _ in range(samples_per_crop):
            n_val = sample_value(*params['N'], is_int=True)
            p_val = sample_value(*params['P'], is_int=True)
            k_val = sample_value(*params['K'], is_int=True)
            temp_val = sample_value(*params['temperature'])
            hum_val = sample_value(*params['humidity'])
            ph_val = sample_value(*params['ph'])
            rain_val = sample_value(*params['rainfall'])

            rows.append([n_val, p_val, k_val, temp_val, hum_val, ph_val, rain_val, crop])

    # Shuffle rows for realistic dataset order
    random.shuffle(rows)

    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)

    print(f"Generated {len(rows)} samples across {len(CROPS_SPEC)} crops at {output_path}")

if __name__ == '__main__':
    target = os.path.join(os.path.dirname(__file__), 'data', 'crop_data.csv')
    generate_csv(target, samples_per_crop=100)
