-- Document Detection Database Schema
-- PostgreSQL initialization script

-- Create extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";
CREATE EXTENSION IF NOT EXISTS "btree_gin";

-- Create custom types
CREATE TYPE document_status AS ENUM (
    'pending',
    'processing',
    'completed',
    'failed',
    'archived'
);

CREATE TYPE document_type AS ENUM (
    'academic',
    'music_score',
    'form',
    'diagram',
    'mixed',
    'unknown'
);

CREATE TYPE processing_stage AS ENUM (
    'uploaded',
    'cleaning',
    'detecting',
    'post_processing',
    'storing',
    'completed'
);

-- Users table for authentication
CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    username VARCHAR(255) UNIQUE NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    is_active BOOLEAN DEFAULT true,
    is_superuser BOOLEAN DEFAULT false,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    last_login TIMESTAMP WITH TIME ZONE,
    metadata JSONB DEFAULT '{}'::jsonb
);

-- API keys table
CREATE TABLE IF NOT EXISTS api_keys (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    key_hash VARCHAR(255) UNIQUE NOT NULL,
    name VARCHAR(255) NOT NULL,
    scopes TEXT[] DEFAULT ARRAY['read', 'write'],
    rate_limit INTEGER DEFAULT 1000,
    expires_at TIMESTAMP WITH TIME ZONE,
    last_used_at TIMESTAMP WITH TIME ZONE,
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB DEFAULT '{}'::jsonb
);

-- Organizations table
CREATE TABLE IF NOT EXISTS organizations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) UNIQUE NOT NULL,
    domain VARCHAR(255),
    is_active BOOLEAN DEFAULT true,
    settings JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- User-Organization relationship
CREATE TABLE IF NOT EXISTS user_organizations (
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    organization_id UUID REFERENCES organizations(id) ON DELETE CASCADE,
    role VARCHAR(50) DEFAULT 'member',
    joined_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (user_id, organization_id)
);

-- Documents table
CREATE TABLE IF NOT EXISTS documents (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    organization_id UUID REFERENCES organizations(id) ON DELETE SET NULL,
    original_filename VARCHAR(500) NOT NULL,
    file_size BIGINT NOT NULL,
    mime_type VARCHAR(100) NOT NULL,
    storage_path VARCHAR(1000) NOT NULL,
    processed_path VARCHAR(1000),
    thumbnail_path VARCHAR(1000),
    status document_status DEFAULT 'pending',
    document_type document_type DEFAULT 'unknown',
    page_count INTEGER DEFAULT 1,
    processing_time_ms INTEGER,
    error_message TEXT,
    metadata JSONB DEFAULT '{}'::jsonb,
    quality_metrics JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    processed_at TIMESTAMP WITH TIME ZONE
);

-- Detection results table
CREATE TABLE IF NOT EXISTS detection_results (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    page_number INTEGER DEFAULT 1,
    detection_count INTEGER DEFAULT 0,
    confidence_avg FLOAT,
    processing_time_ms INTEGER,
    model_version VARCHAR(50),
    detections JSONB NOT NULL DEFAULT '[]'::jsonb,
    metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Processing jobs table
CREATE TABLE IF NOT EXISTS processing_jobs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    job_type VARCHAR(50) NOT NULL,
    stage processing_stage DEFAULT 'uploaded',
    celery_task_id VARCHAR(255) UNIQUE,
    status VARCHAR(50) DEFAULT 'pending',
    priority INTEGER DEFAULT 5,
    retry_count INTEGER DEFAULT 0,
    max_retries INTEGER DEFAULT 3,
    error_message TEXT,
    started_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB DEFAULT '{}'::jsonb
);

-- Audit log table
CREATE TABLE IF NOT EXISTS audit_logs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE SET NULL,
    action VARCHAR(100) NOT NULL,
    resource_type VARCHAR(50),
    resource_id UUID,
    ip_address INET,
    user_agent TEXT,
    request_id UUID,
    changes JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Performance metrics table
CREATE TABLE IF NOT EXISTS performance_metrics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    endpoint VARCHAR(255) NOT NULL,
    method VARCHAR(10) NOT NULL,
    status_code INTEGER,
    response_time_ms INTEGER,
    user_id UUID REFERENCES users(id) ON DELETE SET NULL,
    api_key_id UUID REFERENCES api_keys(id) ON DELETE SET NULL,
    request_size BIGINT,
    response_size BIGINT,
    error BOOLEAN DEFAULT false,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Model versions table
CREATE TABLE IF NOT EXISTS model_versions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL,
    version VARCHAR(50) NOT NULL,
    model_type VARCHAR(100) NOT NULL,
    file_path VARCHAR(1000) NOT NULL,
    accuracy FLOAT,
    parameters JSONB DEFAULT '{}'::jsonb,
    is_active BOOLEAN DEFAULT true,
    is_default BOOLEAN DEFAULT false,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB DEFAULT '{}'::jsonb,
    UNIQUE(name, version)
);

-- Feedback table for model improvement
CREATE TABLE IF NOT EXISTS feedback (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    detection_result_id UUID REFERENCES detection_results(id) ON DELETE CASCADE,
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    rating INTEGER CHECK (rating >= 1 AND rating <= 5),
    correct_detections INTEGER,
    missed_detections INTEGER,
    false_positives INTEGER,
    comments TEXT,
    corrections JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Cache table for expensive operations
CREATE TABLE IF NOT EXISTS cache_entries (
    key VARCHAR(500) PRIMARY KEY,
    value JSONB NOT NULL,
    expires_at TIMESTAMP WITH TIME ZONE NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    access_count INTEGER DEFAULT 0,
    last_accessed TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for performance
CREATE INDEX idx_documents_user_id ON documents(user_id);
CREATE INDEX idx_documents_status ON documents(status);
CREATE INDEX idx_documents_created_at ON documents(created_at);
CREATE INDEX idx_documents_organization_id ON documents(organization_id);
CREATE INDEX idx_documents_document_type ON documents(document_type);

CREATE INDEX idx_detection_results_document_id ON detection_results(document_id);
CREATE INDEX idx_detection_results_created_at ON detection_results(created_at);

CREATE INDEX idx_processing_jobs_document_id ON processing_jobs(document_id);
CREATE INDEX idx_processing_jobs_status ON processing_jobs(status);
CREATE INDEX idx_processing_jobs_celery_task_id ON processing_jobs(celery_task_id);

CREATE INDEX idx_api_keys_user_id ON api_keys(user_id);
CREATE INDEX idx_api_keys_key_hash ON api_keys(key_hash);

CREATE INDEX idx_audit_logs_user_id ON audit_logs(user_id);
CREATE INDEX idx_audit_logs_created_at ON audit_logs(created_at);
CREATE INDEX idx_audit_logs_action ON audit_logs(action);

CREATE INDEX idx_performance_metrics_endpoint ON performance_metrics(endpoint);
CREATE INDEX idx_performance_metrics_created_at ON performance_metrics(created_at);
CREATE INDEX idx_performance_metrics_user_id ON performance_metrics(user_id);

CREATE INDEX idx_cache_entries_expires_at ON cache_entries(expires_at);

-- GIN indexes for JSONB columns
CREATE INDEX idx_documents_metadata_gin ON documents USING gin(metadata);
CREATE INDEX idx_detection_results_detections_gin ON detection_results USING gin(detections);

-- Text search indexes
CREATE INDEX idx_documents_filename_trgm ON documents USING gin(original_filename gin_trgm_ops);

-- Create materialized views
CREATE MATERIALIZED VIEW IF NOT EXISTS user_statistics AS
SELECT 
    u.id as user_id,
    u.username,
    COUNT(DISTINCT d.id) as total_documents,
    COUNT(DISTINCT dr.id) as total_detections,
    AVG(d.processing_time_ms) as avg_processing_time,
    SUM(d.file_size) as total_storage_used,
    MAX(d.created_at) as last_upload
FROM users u
LEFT JOIN documents d ON u.id = d.user_id
LEFT JOIN detection_results dr ON d.id = dr.document_id
GROUP BY u.id, u.username;

CREATE INDEX idx_user_statistics_user_id ON user_statistics(user_id);

-- Create triggers
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_users_updated_at BEFORE UPDATE ON users
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_documents_updated_at BEFORE UPDATE ON documents
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_organizations_updated_at BEFORE UPDATE ON organizations
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Function to clean expired cache entries
CREATE OR REPLACE FUNCTION clean_expired_cache()
RETURNS void AS $$
BEGIN
    DELETE FROM cache_entries WHERE expires_at < CURRENT_TIMESTAMP;
END;
$$ LANGUAGE plpgsql;

-- Create default admin user (password: admin123)
INSERT INTO users (username, email, password_hash, is_superuser)
VALUES (
    'admin',
    'admin@example.com',
    '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewKyNiGcuNjCrZ9y',
    true
) ON CONFLICT (username) DO NOTHING;

-- Create default model versions
INSERT INTO model_versions (name, version, model_type, file_path, is_default, parameters)
VALUES 
    ('yolo_document', 'v8.1.0', 'detection', '/models/yolov8m.pt', true, '{"classes": 26}'::jsonb),
    ('music_detector', 'v1.0.0', 'specialized', '/models/music_detector.pt', false, '{"classes": 6}'::jsonb),
    ('form_detector', 'v1.0.0', 'specialized', '/models/form_detector.pt', false, '{"classes": 4}'::jsonb),
    ('deskew_net', 'v1.0.0', 'preprocessing', '/models/deskew_net.pth', true, '{"angle_range": 45}'::jsonb)
ON CONFLICT (name, version) DO NOTHING;

-- Grant permissions
GRANT ALL ON ALL TABLES IN SCHEMA public TO detector;
GRANT ALL ON ALL SEQUENCES IN SCHEMA public TO detector;
GRANT ALL ON ALL FUNCTIONS IN SCHEMA public TO detector;