from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from django.core.files.storage import default_storage
import uuid, os
import traceback
from .inference import Command


class PredictAPI(APIView):
    def post(self, request):
        file = request.FILES.get('file')
        if not file or not file.name.endswith('.csv'):
            return Response({'error': '请上传 csv'}, status=status.HTTP_400_BAD_REQUEST)

        tmp = f'tmp_{uuid.uuid4().hex}.csv'
        path = default_storage.save(tmp, file)

        try:
            print(f"开始处理上传文件: {file.name}")

            # 调用Command类的预测方法
            df = Command.predict_csv(default_storage.path(path))

            # 转换为前端需要的格式
            data = df[['x', 'y', 'z', 'label']].to_dict(orient='records')

            # 统计信息
            label_counts = df['label'].value_counts().sort_index().to_dict()
            print(f"预测完成! 标签统计: {label_counts}")

            return Response({
                'points': data,
                'statistics': label_counts,
                'message': f'成功处理 {len(df)} 个点'
            }, status=status.HTTP_200_OK)

        except Exception as e:
            print(f"预测出错: {str(e)}")
            print(f"详细错误信息: {traceback.format_exc()}")
            return Response({
                'error': f'预测失败: {str(e)}',
                'detail': str(e)  # 添加详细错误信息
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        finally:
            # 清理临时文件
            if default_storage.exists(path):
                default_storage.delete(path)