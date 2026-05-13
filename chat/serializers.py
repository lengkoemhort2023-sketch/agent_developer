from base.admins import document
from rest_framework import serializers
from .models import ChatSession, ChatMessage, ChatInput
from user.serializers import UserSerializer
from document.models import Document
import json
import uuid
# Chat Input
class ChatInputDetailSerializer(serializers.ModelSerializer):
    voice_file = serializers.SerializerMethodField()

    def get_voice_file(self, obj):
        if not obj.voice_file:
            return None

        try:
            voice_url = obj.voice_file.url
        except ValueError:
            voice_url = None

        if voice_url:
            if voice_url.startswith(("http://", "https://")):
                return voice_url

            normalized_url = voice_url if voice_url.startswith("/") else f"/{voice_url}"
            if normalized_url.startswith("/media/protected/"):
                return normalized_url

        voice_name = getattr(obj.voice_file, "name", "")
        if not voice_name:
            return None

        normalized_name = voice_name.lstrip("/")
        return f"/media/protected/{normalized_name}"

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
    combined_answer = serializers.SerializerMethodField()
    document_references = serializers.SerializerMethodField()

    def get_input_data(self, obj):
        if hasattr(obj, 'input_data') and obj.input_data:
            return ChatInputDetailSerializer(obj.input_data).data
        return None

    def get_document_references(self, obj):
        """Extract document_references from the raw answer data if available"""
        import logging
        logger = logging.getLogger(__name__)
        
        # Retrieve stored model field if available but prefer per-answer refs from answer payload when present
        try:
            model_refs = getattr(obj, 'document_references', None)
            if isinstance(model_refs, str):
                try:
                    model_refs = json.loads(model_refs)
                except Exception:
                    logger.warning("Failed to parse document_references stored as string on model; will fall back to answer parsing")
        except Exception as e:
            logger.error(f"Error accessing model field document_references: {e}", exc_info=True)
            model_refs = None

        if not obj.answer:
            logger.warning("No answer field in ChatMessage")
            return []
        
        def format_page_range(pages):
            if not pages:
                return "N/A"
            try:
                pages = sorted(set(int(p) for p in pages if p is not None))
            except Exception:
                pages = sorted(set(p for p in pages if isinstance(p, int)))
            if not pages:
                return "N/A"
            if len(pages) == 1:
                return f"page {pages[0]}"
            ranges = []
            start = pages[0]
            end = pages[0]
            for p in pages[1:]:
                if p == end + 1:
                    end = p
                else:
                    if start == end:
                        ranges.append(str(start))
                    else:
                        ranges.append(f"{start}-{end}")
                    start = end = p
            if start == end:
                ranges.append(str(start))
            else:
                ranges.append(f"{start}-{end}")
            return f"pages {', '.join(ranges)}"

        try:
            answer_data = obj.answer
            logger.info(f"Raw answer type: {type(answer_data)}, length: {len(str(answer_data))}")
            
            if isinstance(answer_data, str):
                logger.info("Parsing JSON string answer...")
                answer_data = json.loads(answer_data)
                logger.info(f"Parsed answer_data keys: {answer_data.keys() if isinstance(answer_data, dict) else 'not a dict'}")


            # Preserve per-answer document_reference entries (if present) to keep refresh identical to streaming
            if isinstance(answer_data, dict) and isinstance(answer_data.get('answers'), list):
                answers_arr = answer_data.get('answers', [])
                per_answer_refs = []
                for ans in answers_arr:
                    ref = None
                    if isinstance(ans, dict):
                        # Prefer nested document_reference (single) or document_references (list)
                        dr = ans.get('document_reference') if 'document_reference' in ans else ans.get('document_references')
                        if isinstance(dr, dict):
                            ref = dr.copy()
                            if 'html' not in ref:
                                ref['html'] = dr.get('html') or ans.get('html') or ""
                        elif isinstance(dr, list) and dr:
                            # take first reference when a list is provided
                            first_dr = dr[0]
                            if isinstance(first_dr, dict):
                                ref = first_dr.copy()
                                if 'html' not in ref:
                                    ref['html'] = first_dr.get('html') or dr[0].get('html') or ans.get('html') or ""
                        else:
                            # Fallback to common answer-level doc fields
                            doc_id = ans.get('doc_id') or ans.get('file_id')
                            file_name = ans.get('file_name') or ans.get('fileName') or ans.get('title') or ""
                            # Also check for score in the answer itself as fallback
                            answer_score = ans.get('score') or ans.get('max_score') or 0.0
                            if doc_id or file_name:
                                ref = {
                                    'doc_id': doc_id,
                                    'file_name': file_name,
                                    'file_url': ans.get('file_url') or ans.get('fileUrl') or (f"/v/{doc_id}" if doc_id else None),
                                    'chunks_count': 1,
                                    'max_score': float(answer_score),
                                    'pages': ans.get('pages') if isinstance(ans.get('pages'), list) else ([ans.get('page')] if ans.get('page') is not None else []),
                                    'html': ans.get('html') or "",
                                }
                    if ref:
                        # Normalize pages to ints when possible
                        try:
                            if 'pages' in ref and ref['pages'] is not None:
                                ref['pages'] = [int(p) for p in ref['pages'] if p is not None]
                        except Exception:
                            pass
                        
                        # Add page_display and scores_str if not already present
                        if 'page_display' not in ref and 'pages' in ref:
                            ref['page_display'] = format_page_range(ref['pages'])
                        # Add scores_str if not present - try to get max_score from various places
                        if 'scores_str' not in ref:
                            # Try to get max_score from various possible fields
                            max_score = ref.get('max_score') or ref.get('score')
                            if max_score is not None:
                                try:
                                    max_score_float = float(max_score)
                                    if max_score_float >= 0:
                                        ref['scores_str'] = f"{max_score_float:.2f}"
                                    else:
                                        ref['scores_str'] = "N/A"
                                except (TypeError, ValueError):
                                    ref['scores_str'] = "N/A"
                            else:
                                ref['scores_str'] = "N/A"
                        
                        per_answer_refs.append(ref)
                    else:
                        per_answer_refs.append(None)
                # If any per-answer reference exists, return them preserving order (filtering out None)
                if any(r is not None for r in per_answer_refs):
                    cleaned = [r for r in per_answer_refs if r is not None]
                    logger.info(f"✓ Returning {len(cleaned)} per-answer document_references preserved in order")
                    return cleaned

            # If no per-answer references found, but top-level document_references exist, return them (fallback)
            if isinstance(answer_data, dict) and 'document_references' in answer_data:
                refs = answer_data.get('document_references', [])
                for ref in refs:
                    if isinstance(ref, dict):
                        if 'scores_str' not in ref:
                            max_score = ref.get('max_score') or ref.get('score')
                            if max_score is not None:
                                try:
                                    max_score_float = float(max_score)
                                    ref['scores_str'] = f"{max_score_float:.2f}" if max_score_float >= 0 else "N/A"
                                except (TypeError, ValueError):
                                    ref['scores_str'] = "N/A"
                            else:
                                ref['scores_str'] = "N/A"
                        if 'page_display' not in ref:
                            pages = ref.get('pages') or ref.get('page')
                            if pages:
                                if isinstance(pages, list):
                                    ref['page_display'] = format_page_range(pages) if pages else 'N/A'
                                else:
                                    ref['page_display'] = f"page {pages}"
                            else:
                                ref['page_display'] = 'N/A'
                        if 'html' not in ref:
                            ref['html'] = ref.get('html', '') or ""
                logger.info(f"✓ Found top-level document_references fallback: {len(refs)} references")
                return refs

            # Normalize into answers_list if possible
            answers_list = None
            if isinstance(answer_data, dict) and isinstance(answer_data.get('answers'), list):
                answers_list = answer_data.get('answers')
            elif isinstance(answer_data, list):
                answers_list = answer_data

            # If we have an answers list, attempt to build document_references from it
            if isinstance(answers_list, list) and len(answers_list) > 0:
                doc_map = {}
                for item in answers_list:
                    if not isinstance(item, dict):
                        continue
                    doc_id = item.get('doc_id') or item.get('file_id') or item.get('fileId')
                    file_name = item.get('file_name') or item.get('fileName') or item.get('title') or ""
                    pages = []
                    if isinstance(item.get('pages'), list):
                        for p in item.get('pages'):
                            try:
                                pages.append(int(p))
                            except Exception:
                                continue
                    elif item.get('page') is not None:
                        try:
                            pages.append(int(item.get('page')))
                        except Exception:
                            pass
                    max_score = item.get('max_score') or item.get('score') or 0.0

                    if doc_id:
                        entry = doc_map.get(doc_id, {'doc_id': doc_id, 'file_name': file_name or 'Untitled Document', 'file_url': f"/v/{doc_id}", 'chunks_count': 0, 'max_score': 0.0, 'pages': set(), 'html': ''})
                        entry['chunks_count'] = entry.get('chunks_count', 0) + 1
                        try:
                            entry['max_score'] = max(entry.get('max_score', 0.0), float(max_score) if max_score else 0.0)
                        except Exception:
                            entry['max_score'] = entry.get('max_score', 0.0)
                        for p in pages:
                            entry['pages'].add(int(p))
                        if file_name:
                            entry['file_name'] = file_name
                        if item.get('html'):
                            entry['html'] = item.get('html')
                        doc_map[doc_id] = entry

                if doc_map:
                    refs = []
                    for d in doc_map.values():
                        pages_sorted = sorted(list(d['pages']))
                        max_score = float(d['max_score']) if d.get('max_score') is not None else 0.0
                        scores_str = f"{max_score:.2f}"
                        refs.append({
                            'doc_id': d['doc_id'],
                            'file_name': d['file_name'],
                            'file_url': d['file_url'],
                            'chunks_count': d['chunks_count'],
                            'max_score': max_score,
                            'scores_str': scores_str,
                            'pages': pages_sorted,
                            'page_display': format_page_range(pages_sorted) if pages_sorted else 'N/A',
                            'html': d.get('html', ''),
                        })
                    logger.info(f"✓ Built {len(refs)} document_references from answers list")
                    return refs

            # Fallback: document_references embedded inside the first answer entry
            if isinstance(answer_data, dict) and 'answers' in answer_data and isinstance(answer_data.get('answers'), list):
                first = answer_data.get('answers')[0] if answer_data.get('answers') else None
                if isinstance(first, dict) and first.get('document_references'):
                    refs = first.get('document_references')
                    # Add scores_str to each reference
                    for ref in refs:
                        if isinstance(ref, dict) and 'scores_str' not in ref and 'max_score' in ref:
                            try:
                                max_score = float(ref['max_score'])
                                ref['scores_str'] = f"{max_score:.2f}" if max_score >= 0 else "N/A"
                            except (TypeError, ValueError):
                                ref['scores_str'] = "N/A"
                    logger.info(f"✓ Extracted document_references from first answer entry: {len(refs)} refs")
                    return refs

            if isinstance(answer_data, dict):
                logger.warning(f"document_references NOT in answer data. Available keys: {list(answer_data.keys())}")
            else:
                logger.warning(f"answer_data is not a dict/list: {type(answer_data)}")

        except (json.JSONDecodeError, AttributeError, TypeError) as e:
            logger.error(f"Error extracting document_references: {e}", exc_info=True)

        # Fallback to model field if available
        if model_refs:
            try:
                refs = model_refs if isinstance(model_refs, list) else []
                # Add scores_str to each reference
                for ref in refs:
                    if isinstance(ref, dict) and 'scores_str' not in ref and 'max_score' in ref:
                        try:
                            max_score = float(ref['max_score'])
                            ref['scores_str'] = f"{max_score:.2f}" if max_score >= 0 else "N/A"
                        except (TypeError, ValueError):
                            ref['scores_str'] = "N/A"
                logger.info(f"Returning document_references from model field as fallback: {len(refs)} refs")
                return refs
            except Exception:
                pass
            return model_refs if isinstance(model_refs, list) else []

        logger.warning("No document_references found, returning empty list")
        return []

    def get_combined_answer(self, obj):
        """Extract combined_answer with <hr> separators and references from answer data"""
        import logging
        logger = logging.getLogger(__name__)
        
        if not obj.answer:
            return ""
        
        try:
            answer_data = obj.answer
            if isinstance(answer_data, str):
                answer_data = json.loads(answer_data)
            
            # Return combined_answer if it exists in the response
            if isinstance(answer_data, dict) and 'combined_answer' in answer_data:
                combined = answer_data.get('combined_answer', '')
                logger.info(f"✓ Found combined_answer in response, length: {len(combined)}")
                return combined
            
            logger.info("No combined_answer found in response")
            return ""
        except Exception as e:
            logger.error(f"Error extracting combined_answer: {e}")
            return ""

    def get_answer(self, obj):
        answer_data = obj.answer
        if isinstance(answer_data, str):
            # JSON parsing
            try:
                answer_data = json.loads(answer_data)
            except json.JSONDecodeError:
                # Try to fix common JSON issues (single quotes instead of double)
                try:
                    fixed_json = answer_data.replace("'", '"')
                    answer_data = json.loads(fixed_json)
                except json.JSONDecodeError:
                    # If all fail, return a user-friendly error message instead of raw malformed string
                    return "Unable to decode answer. The response may be corrupted."
        
        # Extract answers array regardless of format
        if isinstance(answer_data, dict) and "answers" in answer_data and isinstance(answer_data["answers"], list):
            answers_list = answer_data["answers"]
        elif isinstance(answer_data, list):
            answers_list = answer_data
        else:
            return answer_data
        
        # Enrich answers with published_date
        doc_ids = [item.get('doc_id') for item in answers_list if item.get('doc_id')]
        valid_doc_ids = []
        for doc_id in doc_ids:
            try:
                uuid.UUID(doc_id)
                valid_doc_ids.append(doc_id)
            except (ValueError, TypeError):
                continue
        try:
            docs = Document.objects.filter(id__in=valid_doc_ids)
            doc_map = {str(doc.id): doc for doc in docs}
        except Exception:
            doc_map = {}
        
        return [
            {
                **item,  # Keep all original fields including document_reference
                'file_id': item.get('file_id', '') or item.get('doc_id', ''),
                'published_date': doc_map.get(str(item.get('doc_id'))).published_date if doc_map.get(str(item.get('doc_id'))) else None
            }
            for item in answers_list
        ]

    def get_document(self, obj):
        if hasattr(obj, 'answer') and obj.answer and hasattr(obj.answer, 'file_id'):
            return document.get_document(obj.answer.file_id)
        return None
    
    def get_suggestions(self, obj):
        """Extract suggestions from the answer data"""
        import logging
        logger = logging.getLogger(__name__)
        
        logger.info(f"[DEBUG get_suggestions] obj.id={obj.id}, obj.suggestions={obj.suggestions}, type={type(obj.suggestions)}")
        
        # First check if suggestions are stored directly on the model
        if hasattr(obj, 'suggestions') and obj.suggestions:
            logger.info(f"✅ Found suggestions directly on model: {obj.suggestions}")
            if isinstance(obj.suggestions, list):
                return obj.suggestions
            try:
                parsed = json.loads(obj.suggestions) if isinstance(obj.suggestions, str) else []
                logger.info(f"✅ Parsed suggestions from JSON string: {parsed}")
                return parsed
            except Exception as e:
                logger.warning(f"Failed to parse suggestions JSON: {e}")
                pass
        
        # Fallback: extract from answer JSON
        if not obj.answer:
            logger.warning(f"No answer field for message {obj.id}")
            return []
        
        try:
            answer_data = obj.answer
            if isinstance(answer_data, str):
                answer_data = json.loads(answer_data)
            
            if isinstance(answer_data, dict) and 'suggestions' in answer_data:
                suggestions = answer_data.get('suggestions', [])
                logger.info(f"✅ Extracted suggestions from answer: {suggestions}")
                if isinstance(suggestions, list):
                    return suggestions
        except Exception as e:
            logger.error(f"Error extracting suggestions: {e}")
        
        logger.warning(f"❌ No suggestions found for message {obj.id}")
        return []
    
    suggestions = serializers.SerializerMethodField()
    
    class Meta:
        model = ChatMessage
        fields = ['id', 'session', 'question', 'answer', 'combined_answer', 'sequence', 'created_at', 'input_data', 'document_references', 'suggestions']
        read_only_fields = ['id', 'sequence', 'created_at']

# Chat Session
class ChatSessionDetailSerializer(serializers.ModelSerializer):
    """Detailed serializer with nested messages"""
    user = UserSerializer(read_only=True)
    messages = ChatMessageDetailSerializer(many=True, read_only=True)

    class Meta:
        model = ChatSession
        fields = ['id', 'title', 'user', 'is_archived', 'archived_at', 'created_at', 'updated_at', 'messages']
        read_only_fields = ['id', 'created_at', 'updated_at']

class ChatSessionListSerializer(serializers.ModelSerializer):
    user = UserSerializer(read_only=True)
    message_count = serializers.SerializerMethodField()

    class Meta:
        model = ChatSession
        fields = ['id', 'title', 'user', 'is_archived', 'archived_at', 'created_at', 'updated_at', 'message_count']
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
    document_references = serializers.JSONField(required=False, default=list)
    doc_id = serializers.CharField(required=False, allow_blank=True, allow_null=True)

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






