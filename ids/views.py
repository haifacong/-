import os
import json

import pandas as pd
from django.shortcuts import render
from django.http import JsonResponse, HttpResponse, FileResponse
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings

from .tasks import task_manager

BASE_DIR = settings.BASE_DIR


def index(request):
    """主页面"""
    return render(request, 'ids/index.html')


@csrf_exempt
def api_preprocess(request):
    """数据预处理"""
    if request.method != 'POST':
        return JsonResponse({'error': '仅支持POST'}, status=405)

    data = json.loads(request.body)
    dataset = data.get('dataset', 'cicids2017')
    data_dir = data.get('data_dir', './data')

    preprocess_only = '--preprocess_only' if dataset == 'cicids2017' else ''
    command = f'python data_preprocessing.py --dataset {dataset} --data_dir {data_dir} {preprocess_only}'

    ok, msg = task_manager.start_task('数据预处理', command)
    return JsonResponse({'success': ok, 'message': msg})


@csrf_exempt
def api_train(request):
    """模型训练"""
    if request.method != 'POST':
        return JsonResponse({'error': '仅支持POST'}, status=405)

    data = json.loads(request.body)
    dataset = data.get('dataset', 'cicids2017')
    model = data.get('model', 'cnn')
    batch_size = data.get('batch_size', 64)
    epochs = data.get('epochs', 30)
    lr = data.get('lr', 0.001)
    hidden_dim = data.get('hidden_dim', 128)
    num_layers = data.get('num_layers', 2)
    no_cuda = '--no_cuda' if data.get('no_cuda', False) else ''

    command = (
        f'python train.py --dataset {dataset} --model {model} '
        f'--batch_size {batch_size} --epochs {epochs} --lr {lr} '
        f'--hidden_dim {hidden_dim} --num_layers {num_layers} {no_cuda}'
    )

    def on_complete():
        possible = [
            os.path.join(BASE_DIR, 'training_history.png'),
            os.path.join(BASE_DIR, 'results', f'{model}_{dataset}', 'training_history.png'),
        ]
        for p in possible:
            if os.path.exists(p):
                task_manager.result_data['training_image'] = os.path.relpath(p, BASE_DIR)
                break

    ok, msg = task_manager.start_task('模型训练', command, on_complete=on_complete)
    return JsonResponse({'success': ok, 'message': msg})


@csrf_exempt
def api_evaluate(request):
    """模型评估"""
    if request.method != 'POST':
        return JsonResponse({'error': '仅支持POST'}, status=405)

    data = json.loads(request.body)
    dataset = data.get('dataset', 'cicids2017')
    model = data.get('model', 'cnn')
    batch_size = data.get('batch_size', 64)
    hidden_dim = data.get('hidden_dim', 128)
    num_layers = data.get('num_layers', 2)
    no_cuda = '--no_cuda' if data.get('no_cuda', False) else ''
    data_dir = data.get('data_dir', './data')

    results_dir = f'./results/{model}_{dataset}'
    os.makedirs(results_dir, exist_ok=True)

    command = (
        f'python evaluate.py --dataset {dataset} --model {model} '
        f'--batch_size {batch_size} --hidden_dim {hidden_dim} '
        f'--num_layers {num_layers} --data_dir {data_dir} '
        f'--save_dir {results_dir} {no_cuda}'
    )

    def on_complete():
        task_manager.result_data['results_dir'] = results_dir

    ok, msg = task_manager.start_task('模型评估', command, on_complete=on_complete)
    return JsonResponse({'success': ok, 'message': msg})


@csrf_exempt
def api_capture(request):
    """流量捕获"""
    if request.method != 'POST':
        return JsonResponse({'error': '仅支持POST'}, status=405)

    data = json.loads(request.body)
    capture_time = data.get('capture_time', 60)
    data_dir = data.get('data_dir', './data')

    capture_dir = os.path.join(data_dir, 'capture_flows')
    os.makedirs(capture_dir, exist_ok=True)
    capture_file = os.path.join(capture_dir, 'captured_flows.csv')

    command = f'python main.py --task capture --capture_time {capture_time} --capture_file {capture_file}'

    def on_complete():
        if os.path.exists(capture_file):
            task_manager.result_data['capture_file'] = capture_file

    ok, msg = task_manager.start_task('流量捕获', command, on_complete=on_complete)
    return JsonResponse({'success': ok, 'message': msg})


@csrf_exempt
def api_detect(request):
    """威胁检测"""
    if request.method != 'POST':
        return JsonResponse({'error': '仅支持POST'}, status=405)

    data = json.loads(request.body)
    dataset = data.get('dataset', 'cicids2017')
    model = data.get('model', 'cnn')
    no_cuda = '--no_cuda' if data.get('no_cuda', False) else ''
    data_dir = data.get('data_dir', './data')

    capture_file = os.path.join(data_dir, 'capture_flows', 'captured_flows.csv')
    if not os.path.exists(capture_file):
        return JsonResponse({'success': False, 'message': f'捕获文件不存在: {capture_file}'})

    command = (
        f'python main.py --task detect --capture_file {capture_file} '
        f'--detect_model {model} --dataset {dataset} {no_cuda}'
    )

    def on_complete():
        results_dir = os.path.join(BASE_DIR, 'results')
        if os.path.exists(results_dir):
            result_files = [
                f for f in os.listdir(results_dir)
                if f.startswith('detection_result_') and f.endswith('.csv')
            ]
            if result_files:
                latest = max(result_files, key=lambda x: os.path.getmtime(os.path.join(results_dir, x)))
                task_manager.result_data['detection_file'] = os.path.join('results', latest)

    ok, msg = task_manager.start_task('威胁检测', command, on_complete=on_complete)
    return JsonResponse({'success': ok, 'message': msg})


def api_task_status(request):
    """获取任务状态和新日志（轮询接口）"""
    since = int(request.GET.get('since', 0))
    status = task_manager.get_status()
    new_logs = task_manager.get_logs(since)
    status['new_logs'] = new_logs
    status['log_offset'] = since + len(new_logs)
    return JsonResponse(status)


@csrf_exempt
def api_export_log(request):
    """导出日志"""
    logs = task_manager.get_logs()
    content = '\n'.join(logs)
    response = HttpResponse(content, content_type='text/plain; charset=utf-8')
    response['Content-Disposition'] = 'attachment; filename="ids_log.txt"'
    return response


def api_result_image(request, image_name):
    """获取结果图片"""
    dataset = request.GET.get('dataset', 'cicids2017')
    model = request.GET.get('model', 'cnn')

    possible_paths = [
        os.path.join(BASE_DIR, image_name),
        os.path.join(BASE_DIR, 'results', f'{model}_{dataset}', image_name),
        os.path.join(BASE_DIR, 'results', image_name),
    ]

    for path in possible_paths:
        if os.path.exists(path):
            return FileResponse(open(path, 'rb'), content_type='image/png')

    return JsonResponse({'error': f'图片未找到: {image_name}'}, status=404)


def api_capture_data(request):
    """获取捕获流量表格数据"""
    data_dir = request.GET.get('data_dir', './data')
    capture_file = os.path.join(data_dir, 'capture_flows', 'captured_flows.csv')

    if not os.path.exists(capture_file):
        return JsonResponse({'error': '无捕获数据'}, status=404)

    try:
        df = pd.read_csv(capture_file)
        df_display = df.head(100)
        return JsonResponse({
            'columns': list(df.columns),
            'data': df_display.values.tolist(),
            'total': len(df),
        })
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)


def api_detection_result(request):
    """获取检测结果表格数据"""
    results_dir = os.path.join(BASE_DIR, 'results')
    if not os.path.exists(results_dir):
        return JsonResponse({'error': '无检测结果'}, status=404)

    result_files = [
        f for f in os.listdir(results_dir)
        if f.startswith('detection_result_') and f.endswith('.csv')
    ]
    if not result_files:
        return JsonResponse({'error': '无检测结果'}, status=404)

    latest = max(result_files, key=lambda x: os.path.getmtime(os.path.join(results_dir, x)))
    filepath = os.path.join(results_dir, latest)

    try:
        df = pd.read_csv(filepath)
        df_display = df.head(100)

        stats = {'total': len(df)}
        attack_cols = ['is_attack', 'attack', 'Attack', 'attack_type', 'label']
        for col in attack_cols:
            if col in df.columns:
                try:
                    if df[col].dtype in ['int64', 'float64', 'bool']:
                        attack_count = int(df[col].sum())
                    else:
                        attack_count = len(df[df[col].astype(str).str.lower() != 'normal'])
                    stats['attack_count'] = attack_count
                    stats['normal_count'] = len(df) - attack_count
                    stats['attack_ratio'] = round(attack_count / len(df) * 100, 2) if len(df) > 0 else 0
                except Exception:
                    pass
                break

        return JsonResponse({
            'columns': list(df.columns),
            'data': df_display.values.tolist(),
            'stats': stats,
        })
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)
