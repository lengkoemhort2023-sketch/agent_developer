from django.db import models
from user.models import User
import uuid

class ChatSession(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    title = models.CharField(max_length=255, null=True, blank=True)
    user = models.ForeignKey(User, on_delete=models.SET_NULL, null=True, blank=True)
    is_archived = models.BooleanField(default=False)
    archived_at = models.DateTimeField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        db_table = 'chat_sessions'
        indexes = [
            models.Index(
                fields=['user', 'is_archived', '-updated_at'],
                name='chat_sessio_user_id_64cd64_idx',
            ),
        ]
    
    def __str__(self):
        return f"Chat Session {self.id}"
    
    @classmethod
    def get_last_empty(cls, user=None):
        qs = cls.objects.all()
        if user:
            qs = qs.filter(user=user, is_archived=False)
        else:
            return None
        qs = qs.annotate(msg_count=models.Count('messages')).filter(msg_count=0)
        return qs.order_by('-created_at').first()
class ChatMessage(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4)
    session = models.ForeignKey(ChatSession, on_delete=models.CASCADE, related_name='messages')
    question = models.TextField()
    answer = models.TextField()
    document_references = models.JSONField(default=list, blank=True, null=True)
    suggestions = models.JSONField(default=list, blank=True, null=True)
    sequence = models.IntegerField(default=0)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        db_table = 'chat_messages'
        ordering = ['session', 'sequence']
        unique_together = ['session', 'sequence']
    
    def save(self, *args, **kwargs):
        if not self.sequence:
            last_message = ChatMessage.objects.filter(session=self.session).order_by('-sequence').first()
            self.sequence = (last_message.sequence + 1) if last_message else 1
        super().save(*args, **kwargs)
    
    def __str__(self):
        return f"Message {self.sequence} in Session {self.session.id}"

class MessageFeedback(models.Model):
    RATING_POSITIVE = 1
    RATING_NEGATIVE = -1
    RATING_CHOICES = [
        (RATING_POSITIVE, 'Positive'),
        (RATING_NEGATIVE, 'Negative'),
    ]

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    message = models.ForeignKey(
        ChatMessage, on_delete=models.SET_NULL,
        null=True, blank=True, related_name='feedbacks'
    )
    # Store message_id as string fallback in case the ChatMessage is deleted
    message_id_str = models.CharField(max_length=36, null=True, blank=True, db_index=True)
    user = models.ForeignKey(
        User, on_delete=models.SET_NULL,
        null=True, blank=True, related_name='feedbacks'
    )
    rating = models.IntegerField(choices=RATING_CHOICES)  # 1 = thumbs up, -1 = thumbs down
    feedback_text = models.TextField(blank=True, null=True)
    question = models.TextField(blank=True, null=True)
    answer = models.TextField(blank=True, null=True)
    user_name = models.CharField(max_length=255, blank=True, null=True)
    user_email = models.EmailField(blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = 'message_feedbacks'
        ordering = ['-created_at']
        # One feedback record per user per message (upsertable)
        unique_together = [('message_id_str', 'user')]

    def __str__(self):
        rating_label = 'positive' if self.rating == self.RATING_POSITIVE else 'negative'
        return f"Feedback ({rating_label}) on message {self.message_id_str}"


class ChatInput(models.Model):
    INPUT_TYPES = [
        ('text', 'Text Input'),
        ('voice', 'Voice Input'),
    ]
    
    id = models.UUIDField(primary_key=True, default=uuid.uuid4)
    message = models.OneToOneField(ChatMessage, on_delete=models.SET_NULL, null=True, blank=True, related_name='input_data')
    user = models.ForeignKey(User, on_delete=models.SET_NULL, null=True, blank=True)
    input_type = models.CharField(max_length=10, choices=INPUT_TYPES, default='text')
    voice_file = models.FileField(upload_to='voice_inputs/', blank=True, null=True)
    voice_duration = models.FloatField(blank=True, null=True)
    content = models.TextField(blank=True, null=True)
    
    created_at = models.DateTimeField(auto_now_add=True)
    processed_at = models.DateTimeField(blank=True, null=True)
    
    class Meta:
        db_table = 'chat_inputs'
        ordering = ['message__session', 'created_at']
    
    def __str__(self):
        return f"{self.input_type.title()} Input for Message {self.message.id if self.message else 'None'}"
    
    @property
    def is_processed(self):
        """Check if the input has been processed"""
        return self.processed_at is not None





