MANUAL_NAMESPACE = "versioned_manuals"


MANUAL_SPECS = [
    {
        "slug": "orchid_cli",
        "query_subject": "Orchid CLI",
        "update_query": "What command does Orchid CLI use to pull remote worktrees?",
        "retention_query": "What config file does Orchid CLI use?",
        "base_value": "orchid sync",
        "updated_value": "orchid pull",
        "retained_value": "orchid.yml",
    },
    {
        "slug": "nova_build",
        "query_subject": "Nova Build",
        "update_query": "What default cache mode does Nova Build use?",
        "retention_query": "What manifest file does Nova Build use?",
        "base_value": "local",
        "updated_value": "shared_remote",
        "retained_value": "nova.json",
    },
    {
        "slug": "ruma_agent",
        "query_subject": "RUMA Agent",
        "update_query": "What default retrieval top_k does RUMA Agent use?",
        "retention_query": "What answer mode does RUMA Agent use?",
        "base_value": "2",
        "updated_value": "4",
        "retained_value": "citation-first",
    },
    {
        "slug": "cinder_queue",
        "query_subject": "Cinder Queue",
        "update_query": "What default retry backoff mode does Cinder Queue use?",
        "retention_query": "What config file does Cinder Queue use?",
        "base_value": "fixed_30s",
        "updated_value": "jittered_exponential",
        "retained_value": "cinder.toml",
    },
    {
        "slug": "helios_packager",
        "query_subject": "Helios Packager",
        "update_query": "What default publish target does Helios Packager use?",
        "retention_query": "What manifest file does Helios Packager use?",
        "base_value": "internal",
        "updated_value": "staging_registry",
        "retained_value": "helios.yaml",
    },
    {
        "slug": "marlin_proxy",
        "query_subject": "Marlin Proxy",
        "update_query": "What default healthcheck command does Marlin Proxy use?",
        "retention_query": "What config file does Marlin Proxy use?",
        "base_value": "marlin probe",
        "updated_value": "marlin health",
        "retained_value": "proxy.ini",
    },
    {
        "slug": "quartz_scheduler",
        "query_subject": "Quartz Scheduler",
        "update_query": "What default schedule mode does Quartz Scheduler use?",
        "retention_query": "What config file does Quartz Scheduler use?",
        "base_value": "steady",
        "updated_value": "adaptive",
        "retained_value": "quartz.yaml",
    },
    {
        "slug": "atlas_monitor",
        "query_subject": "Atlas Monitor",
        "update_query": "What default alert mode does Atlas Monitor use?",
        "retention_query": "What config file does Atlas Monitor use?",
        "base_value": "summary",
        "updated_value": "adaptive_summary",
        "retained_value": "atlas.toml",
    },
    {
        "slug": "ember_runner",
        "query_subject": "Ember Runner",
        "update_query": "What default queue class does Ember Runner use?",
        "retention_query": "What config file does Ember Runner use?",
        "base_value": "standard",
        "updated_value": "priority_burst",
        "retained_value": "ember.yml",
    },
    {
        "slug": "lumen_cache",
        "query_subject": "Lumen Cache",
        "update_query": "What default eviction policy does Lumen Cache use?",
        "retention_query": "What config file does Lumen Cache use?",
        "base_value": "lru",
        "updated_value": "segmented_lru",
        "retained_value": "lumen.toml",
    },
    {
        "slug": "sable_deploy",
        "query_subject": "Sable Deploy",
        "update_query": "What default rollout mode does Sable Deploy use?",
        "retention_query": "What config file does Sable Deploy use?",
        "base_value": "linear",
        "updated_value": "canary_safe",
        "retained_value": "sable.yaml",
    },
    {
        "slug": "tidal_sync",
        "query_subject": "Tidal Sync",
        "update_query": "What default snapshot mode does Tidal Sync use?",
        "retention_query": "What config file does Tidal Sync use?",
        "base_value": "daily",
        "updated_value": "hourly_delta",
        "retained_value": "tidal.json",
    },
]


CONFLICT_SPECS = [
    {
        "slug": "ruma_agent",
        "conflict_file": "versioned_manual_conflicts/ruma_operator_guide.md",
        "lineage": f"{MANUAL_NAMESPACE}::operator_guide::ruma_agent",
        "query": "What do the RUMA Agent manual and operator guide say about retrieval top_k?",
        "must_contain": "6",
    },
    {
        "slug": "marlin_proxy",
        "conflict_file": "versioned_manual_conflicts/marlin_edge_runbook.md",
        "lineage": f"{MANUAL_NAMESPACE}::operator_guide::marlin_proxy",
        "query": "What do the Marlin Proxy manual and edge runbook say about the default healthcheck command?",
        "must_contain": "marlin health --edge",
    },
    {
        "slug": "quartz_scheduler",
        "conflict_file": "versioned_manual_conflicts/quartz_night_ops.md",
        "lineage": f"{MANUAL_NAMESPACE}::operator_guide::quartz_scheduler",
        "query": "What do the Quartz Scheduler manual and night ops runbook say about the default schedule mode?",
        "must_contain": "burst_safe",
    },
    {
        "slug": "atlas_monitor",
        "conflict_file": "versioned_manual_conflicts/atlas_incident_runbook.md",
        "lineage": f"{MANUAL_NAMESPACE}::operator_guide::atlas_monitor",
        "query": "What do the Atlas Monitor manual and incident runbook say about the default alert mode?",
        "must_contain": "always_page",
    },
    {
        "slug": "sable_deploy",
        "conflict_file": "versioned_manual_conflicts/sable_release_runbook.md",
        "lineage": f"{MANUAL_NAMESPACE}::operator_guide::sable_deploy",
        "query": "What do the Sable Deploy manual and release runbook say about the default rollout mode?",
        "must_contain": "serial_guard",
    },
    {
        "slug": "tidal_sync",
        "conflict_file": "versioned_manual_conflicts/tidal_recovery_runbook.md",
        "lineage": f"{MANUAL_NAMESPACE}::operator_guide::tidal_sync",
        "query": "What do the Tidal Sync manual and recovery runbook say about the default snapshot mode?",
        "must_contain": "full_mirror",
    },
]


BASE_FILES = [f"versioned_manuals/{spec['slug']}_v1.md" for spec in MANUAL_SPECS]
UPDATE_FILES = [
    (f"versioned_manuals/{spec['slug']}_v1.md", f"versioned_manual_updates/{spec['slug']}_v2.md")
    for spec in MANUAL_SPECS
]
CONFLICT_FILES = [
    (spec["conflict_file"], spec["lineage"])
    for spec in CONFLICT_SPECS
]


def build_base_known_cases():
    return [
        {
            "name": f"{spec['slug'].split('_', 1)[0]}_base_value",
            "query": spec["update_query"],
            "must_contain": spec["base_value"],
            "expected_source_suffix": f"versioned_manuals/{spec['slug']}_v1.md",
        }
        for spec in MANUAL_SPECS
    ]


def build_update_cases():
    return [
        {
            "name": f"{spec['slug'].split('_', 1)[0]}_updated_value",
            "query": spec["update_query"],
            "must_contain": spec["updated_value"],
            "must_not_contain": spec["base_value"],
            "expected_source_suffix": f"versioned_manual_updates/{spec['slug']}_v2.md",
        }
        for spec in MANUAL_SPECS
    ]


def build_retention_cases():
    return [
        {
            "name": f"{spec['slug'].split('_', 1)[0]}_retained_value",
            "query": spec["retention_query"],
            "must_contain": spec["retained_value"],
        }
        for spec in MANUAL_SPECS
    ]


def build_conflict_cases():
    return [
        {
            "name": f"{spec['slug'].split('_', 1)[0]}_conflicting_guidance",
            "query": spec["query"],
            "must_contain": spec["must_contain"],
            "expected_source_suffix": spec["conflict_file"],
            "expect_conflict": True,
            "max_sentences": 2,
            "top_k": 5,
        }
        for spec in CONFLICT_SPECS
    ]


BASE_KNOWN_CASES = build_base_known_cases()
UPDATE_CASES = build_update_cases()
RETENTION_CASES = build_retention_cases()
CONFLICT_CASES = build_conflict_cases()
