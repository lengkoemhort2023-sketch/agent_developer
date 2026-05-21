from base.admins import document
from rest_framework import serializers
from .models import ChatSession, ChatMessage, ChatInput
from user.serializers import UserSerializer
from document.models import Document
import ast, json
from base.utils.tonl_helper import tonl_decode
# Chat Input
class ChatInputDetailSerializer(serializers.ModelSerializer):
    class Meta:
        model = ChatInput
        fields = [
            'id','input_type', 'voice_file', 'message',
            'voice_duration', 'content', 'created_at', 'processed_at', 'is_processed'
        ]
        read_only_fields = ['id', 'created_at', 'processed_at']

# Chat Message
class ChatMessageDetailSerializer(serializers.ModelSerializer):
    """Detailed serializer with nested input data and clean up answer"""
    input_data = serializers.SerializerMethodField()
    answer = serializers.SerializerMethodField()
    def get_input_data(self, obj):
        if hasattr(obj, 'input_data') and obj.input_data:
            return ChatInputDetailSerializer(obj.input_data).data
        return None

    def get_answer(self, obj):
        answer_data = obj.answer
        if isinstance(answer_data, str):
            # Try TONL first for token efficiency
            try:
                answer_data = tonl_decode(answer_data)
            except Exception:
                # Fallback to JSON
                try:
                    answer_data = json.loads(answer_data)
                except json.JSONDecodeError:
                    # Try to fix common JSON issues (single quotes instead of double)
                    try:
                        fixed_json = answer_data.replace("'", '"')
                        answer_data = json.loads(fixed_json)
                    except json.JSONDecodeError:
                        # Fallback to literal_eval for older entries
                        try:
                            answer_data = ast.literal_eval(answer_data)
                        except Exception as e:
                            # If all fail, return a user-friendly error message instead of raw malformed string
                            return "Unable to decode answer. The response may be corrupted."
        
        # Ensure answer_data is a list of dictionaries
        if isinstance(answer_data, dict) and "answers" in answer_data and isinstance(answer_data["answers"], list):
            answer_data = answer_data["answers"]
        elif not isinstance(answer_data, list):
            return answer_data # Return raw data if not a list or expected dict format

        if isinstance(answer_data, list):
            doc_ids = [item.get('doc_id') for item in answer_data if item.get('doc_id')]
            docs = Document.objects.filter(id__in=doc_ids)
            doc_map = {str(doc.id): doc for doc in docs}
            return [
                {
                    'text': item.get('text', ''),
                    'file_name': item.get('file_name', ''),
                    'file_id': item.get('file_id', '') or item.get('doc_id', ''),
                    'published_date': doc_map.get(str(item.get('doc_id'))).published_date if doc_map.get(str(item.get('doc_id'))) else None
                }
                for item in answer_data
            ]
        return answer_data

    def get_document(self, obj):
        if hasattr(obj, 'answer') and obj.answer and hasattr(obj.answer, 'file_id'):
            return document.get_document(obj.answer.file_id)
        return None
    class Meta:
        model = ChatMessage
        fields = ['id', 'session', 'question', 'answer', 'sequence', 'created_at', 'input_data']
        read_only_fields = ['id', 'sequence', 'created_at']

# Chat Session
class ChatSessionDetailSerializer(serializers.ModelSerializer):
    """Detailed serializer with nested messages"""
    user = UserSerializer(read_only=True)
    messages = ChatMessageDetailSerializer(many=True, read_only=True)

    class Meta:
        model = ChatSession
        fields = ['id', 'title', 'user', 'created_at', 'updated_at', 'messages']
        read_only_fields = ['id', 'created_at', 'updated_at']

class ChatSessionListSerializer(serializers.ModelSerializer):
    user = UserSerializer(read_only=True)
    message_count = serializers.SerializerMethodField()

    class Meta:
        model = ChatSession
        fields = ['id', 'title', 'user', 'created_at', 'updated_at', 'message_count']
        read_only_fields = ['id', 'created_at', 'updated_at']

    def get_message_count(self, obj):
        return obj.messages.count()

# ============================================================================
# CHAT MESSAGE SERIALIZERS
# ============================================================================

class ChatMessageListSerializer(serializers.ModelSerializer):
    session = ChatSessionListSerializer(read_only=True)
    
    class Meta:
        model = ChatMessage
        fields = ['id', 'session', 'question', 'answer', 'sequence', 'created_at']
        read_only_fields = ['id', 'sequence', 'created_at']

class ChatMessageCreateSerializer(serializers.Serializer):
    """Serializer for creating chat messages or voice input"""
    question = serializers.CharField(required=False, allow_blank=True)
    voice_file = serializers.FileField(required=False, allow_null=True)
    input_type = serializers.ChoiceField(choices=[('text', 'Text'), ('voice', 'Voice')], default='text')
    session = serializers.PrimaryKeyRelatedField(required=False, queryset=ChatSession.objects.all())
    answer = serializers.CharField(required=False, allow_blank=True, allow_null=True)
    document_type = serializers.CharField(required=False, allow_blank=True, allow_null=True)

    def validate(self, data):
        input_type = data.get('input_type', 'text')
        if input_type == 'voice' and not data.get('voice_file'):
            raise serializers.ValidationError("voice_file is required when input_type is 'voice'.")
        if input_type == 'text' and not data.get('question'):
            raise serializers.ValidationError("question is required when input_type is 'text'.")
        return data

# Chat Input
class ChatInputSerializer(serializers.ModelSerializer):
    message = ChatMessageListSerializer(read_only=True)
    user = UserSerializer(read_only=True)
    
    class Meta:
        model = ChatInput
        fields = [
            'id', 'message', 'user', 'input_type', 'content', 
            'created_at', 'is_processed'
        ]
        read_only_fields = ['id', 'created_at']







