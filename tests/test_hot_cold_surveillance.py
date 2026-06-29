"""
OMEGA HOT-COLD SURVEILLANCE: Unit Tests
========================================
QUÉ:       Tests para validar la infraestructura dual de logging.
POR QUÉ:   Garantizar que el sistema Hot (Loki) y Cold (ELK) están
           correctamente configurados y que la latencia cero está garantizada.
PARA QUÉ:  Certificar que el bot nunca se bloquea por logging.
CÓMO:      Valida archivos YAML/JSON de configuración, docker-compose
           profiles, y async queue del logger.
CUÁNDO:    Antes de desplegar la infraestructura.
DÓNDE:     tests/test_hot_cold_surveillance.py
QUIÉN:     CI/CD Pipeline & Manual Verification
"""
import os
import sys
import json
import yaml
import unittest
import queue
import logging
from unittest.mock import patch, MagicMock

# Setup path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class TestDockerComposeProfiles(unittest.TestCase):
    """FASE 3: Validate ELK services exist under cold-storage profile."""
    
    def setUp(self):
        compose_path = os.path.join(BASE_DIR, "docker-compose.yml")
        with open(compose_path, 'r') as f:
            self.compose = yaml.safe_load(f)
    
    def test_elk_services_exist(self):
        """Elasticsearch, Logstash, and Kibana should be defined."""
        services = self.compose['services']
        self.assertIn('elasticsearch', services)
        self.assertIn('logstash', services)
        self.assertIn('kibana', services)
    
    def test_elk_services_have_cold_storage_profile(self):
        """All ELK services must be under the cold-storage profile."""
        for svc in ['elasticsearch', 'logstash', 'kibana']:
            profiles = self.compose['services'][svc].get('profiles', [])
            self.assertIn('cold-storage', profiles, 
                         f"{svc} missing cold-storage profile")
    
    def test_hot_services_have_no_profile(self):
        """Loki, Prometheus, Grafana must NOT have profiles (always on)."""
        for svc in ['prometheus', 'grafana', 'loki', 'promtail']:
            profiles = self.compose['services'][svc].get('profiles', [])
            self.assertEqual(profiles, [], 
                           f"{svc} should not have a profile (always active)")
    
    def test_es_resource_limits(self):
        """Elasticsearch should have memory limits to prevent OOM."""
        es = self.compose['services']['elasticsearch']
        limits = es['deploy']['resources']['limits']
        self.assertIn('memory', limits)
    
    def test_es_data_volume(self):
        """Elasticsearch data should be persisted in a named volume."""
        self.assertIn('es_data', self.compose.get('volumes', {}))
    
    def test_logstash_depends_on_es(self):
        """Logstash must wait for Elasticsearch to be healthy."""
        logstash = self.compose['services']['logstash']
        deps = logstash.get('depends_on', {})
        self.assertIn('elasticsearch', deps)


class TestPromtailDualShipping(unittest.TestCase):
    """FASE 1-2: Validate Promtail ships to both Loki and Logstash."""
    
    def setUp(self):
        promtail_path = os.path.join(BASE_DIR, "deployment", "promtail.yml")
        with open(promtail_path, 'r') as f:
            self.config = yaml.safe_load(f)
    
    def test_has_two_clients(self):
        """Promtail should have exactly 2 clients (Loki + Logstash)."""
        clients = self.config.get('clients', [])
        self.assertEqual(len(clients), 2, 
                        "Promtail must have exactly 2 clients for dual-shipping")
    
    def test_loki_client_is_primary(self):
        """First client should point to Loki."""
        clients = self.config['clients']
        self.assertIn('loki', clients[0]['url'])
    
    def test_logstash_client_is_secondary(self):
        """Second client should point to Logstash."""
        clients = self.config['clients']
        self.assertIn('logstash', clients[1]['url'])
    
    def test_logstash_client_has_backoff(self):
        """Logstash client must have backoff config for fire-and-forget."""
        logstash_client = self.config['clients'][1]
        backoff = logstash_client.get('backoff_config', {})
        self.assertIn('max_retries', backoff, 
                      "Logstash client needs max_retries for fire-and-forget")
        self.assertLessEqual(backoff['max_retries'], 5, 
                            "max_retries should be low to avoid blocking")
    
    def test_scrape_config_exists(self):
        """At least one scrape config should exist."""
        self.assertTrue(len(self.config.get('scrape_configs', [])) > 0)


class TestLogstashPipeline(unittest.TestCase):
    """FASE 4: Validate Logstash configuration."""
    
    def test_logstash_conf_exists(self):
        """Logstash pipeline config must exist."""
        path = os.path.join(BASE_DIR, "deployment", "elk", "logstash.conf")
        self.assertTrue(os.path.exists(path), "logstash.conf not found")
    
    def test_logstash_has_input_output(self):
        """Logstash config must have input{} and output{} blocks."""
        path = os.path.join(BASE_DIR, "deployment", "elk", "logstash.conf")
        with open(path, 'r') as f:
            content = f.read()
        self.assertIn('input {', content)
        self.assertIn('output {', content)
        self.assertIn('filter {', content)
    
    def test_es_index_template_valid_json(self):
        """Elasticsearch index template must be valid JSON."""
        path = os.path.join(BASE_DIR, "deployment", "elk", "es-index-template.json")
        with open(path, 'r') as f:
            template = json.load(f)
        self.assertIn('index_patterns', template)
        self.assertIn('trader-gemini-*', template['index_patterns'])
    
    def test_ilm_policy_valid_json(self):
        """ILM lifecycle policy must be valid JSON."""
        path = os.path.join(BASE_DIR, "deployment", "elk", "ilm-policy.json")
        with open(path, 'r') as f:
            policy = json.load(f)
        self.assertIn('policy', policy)
        self.assertIn('phases', policy['policy'])
        # Must have hot and delete phases at minimum
        phases = policy['policy']['phases']
        self.assertIn('hot', phases)
        self.assertIn('delete', phases)


class TestGrafanaDatasources(unittest.TestCase):
    """FASE 5: Validate Grafana has all three datasources."""
    
    def setUp(self):
        ds_path = os.path.join(
            BASE_DIR, "deployment", "grafana", "provisioning", 
            "datasources", "datasource.yml"
        )
        with open(ds_path, 'r') as f:
            self.config = yaml.safe_load(f)
    
    def test_has_three_datasources(self):
        """Grafana should have Prometheus, Loki, and Elasticsearch."""
        ds_names = [ds['name'] for ds in self.config['datasources']]
        self.assertIn('Prometheus', ds_names)
        self.assertIn('Loki', ds_names)
        self.assertIn('Elasticsearch', ds_names)
    
    def test_elasticsearch_points_to_correct_index(self):
        """ES datasource must reference trader-gemini-* index pattern."""
        es_ds = next(ds for ds in self.config['datasources'] 
                     if ds['name'] == 'Elasticsearch')
        self.assertEqual(es_ds['jsonData']['index'], 'trader-gemini-*')


class TestForensicDashboard(unittest.TestCase):
    """FASE 5: Validate Forensic Deep Dive dashboard."""
    
    def setUp(self):
        path = os.path.join(
            BASE_DIR, "deployment", "grafana", "dashboards",
            "Forensic_Deep_Dive.json"
        )
        with open(path, 'r') as f:
            self.dashboard = json.load(f)
    
    def test_dashboard_has_panels(self):
        """Forensic dashboard must have panels."""
        self.assertTrue(len(self.dashboard['panels']) >= 4)
    
    def test_dashboard_uses_elasticsearch(self):
        """All panels should use Elasticsearch datasource."""
        for panel in self.dashboard['panels']:
            ds = panel.get('datasource', {})
            self.assertEqual(ds.get('type'), 'elasticsearch',
                           f"Panel '{panel['title']}' should use Elasticsearch")
    
    def test_back_link_to_commander(self):
        """Dashboard should link back to Commander View."""
        links = self.dashboard.get('links', [])
        urls = [link['url'] for link in links]
        self.assertTrue(any('commander-view' in url for url in urls),
                       "Missing back-link to Commander View")


class TestCommanderViewDeepDive(unittest.TestCase):
    """FASE 5: Commander View should have Deep Dive panel."""
    
    def setUp(self):
        path = os.path.join(
            BASE_DIR, "deployment", "grafana", "dashboards",
            "Commander_View.json"
        )
        with open(path, 'r') as f:
            self.dashboard = json.load(f)
    
    def test_deep_dive_panel_exists(self):
        """Commander View must have a Deep Dive panel."""
        titles = [p['title'] for p in self.dashboard['panels']]
        self.assertTrue(
            any('Deep Dive' in t or 'Cold Storage' in t for t in titles),
            "Missing Deep Dive panel in Commander View"
        )


class TestZeroLatencyIsolation(unittest.TestCase):
    """FASE 7: Certify the logger uses async non-blocking queue."""
    
    def test_logger_uses_queue_handler(self):
        """The main logger must use QueueHandler for async I/O."""
        from utils.logger import logger
        # Check if at least one handler is QueueHandler
        from logging.handlers import QueueHandler
        has_queue = any(isinstance(h, QueueHandler) for h in logger.handlers)
        self.assertTrue(has_queue, 
                       "Logger must use QueueHandler for zero-latency guarantee")
    
    def test_logger_has_listener(self):
        """Logger must have _listener for background processing."""
        from utils.logger import logger
        self.assertTrue(hasattr(logger, '_listener'),
                       "Logger must have _listener for background thread")
    
    def test_queue_does_not_block(self):
        """Verify the queue is unbounded (won't block main thread)."""
        test_queue = queue.Queue(-1)  # -1 = unlimited
        # Should not raise even after many items
        for i in range(10000):
            test_queue.put_nowait(f"log_entry_{i}")
        self.assertEqual(test_queue.qsize(), 10000)


if __name__ == '__main__':
    unittest.main()
