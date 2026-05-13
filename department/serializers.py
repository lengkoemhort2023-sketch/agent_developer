from rest_framework import serializers
from .models import Department

class DepartmentSerializer(serializers.ModelSerializer):
    """Basic serializer for Department model"""
    
    class Meta:
        model = Department
        fields = ['id', 'name', 'created_at', 'updated_at']
        read_only_fields = ['id', 'created_at', 'updated_at']

class DepartmentCreateSerializer(serializers.ModelSerializer):
    """Serializer for creating departments"""
    
    class Meta:
        model = Department
        fields = ['name']

class DepartmentUpdateSerializer(serializers.ModelSerializer):
    """Serializer for updating departments"""
    
    class Meta:
        model = Department
        fields = ['name']

class DepartmentListSerializer(serializers.ModelSerializer):
    """Serializer for listing departments with minimal data"""
    
    class Meta:
        model = Department
        fields = ['id', 'name'] 







