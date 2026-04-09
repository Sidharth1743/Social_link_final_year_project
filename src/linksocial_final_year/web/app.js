const state = {
  overview: null,
  selectedProfile: null,
  selectedTarget: "instagram",
  selectedModel: "semantic_hybrid_gbdt",
  selectedCandidateId: null,
  lastLinkResponse: null,
};

const elements = {
  heroStats: document.getElementById("hero-stats"),
  profileList: document.getElementById("profile-list"),
  platformFilter: document.getElementById("platform-filter"),
  profileQuery: document.getElementById("profile-query"),
  searchButton: document.getElementById("search-button"),
  targetPlatform: document.getElementById("target-platform"),
  modelSelect: document.getElementById("model-select"),
  linkButton: document.getElementById("link-button"),
  inspector: document.getElementById("inspector-content"),
  pairwiseTable: document.getElementById("pairwise-table"),
  multiTable: document.getElementById("multi-table"),
  graphSvg: document.getElementById("graph-svg"),
};

async function api(path) {
  const response = await fetch(path);
  if (!response.ok) {
    const payload = await response.json().catch(() => ({ detail: "Request failed" }));
    throw new Error(payload.detail || "Request failed");
  }
  return response.json();
}

function platformLabel(platform) {
  return platform.replace("_", "+").replace("google+", "google+").replace("google_plus", "google+");
}

function badge(platform) {
  return `<span class="badge ${platform}">${platformLabel(platform)}</span>`;
}

function formatNumber(value) {
  return typeof value === "number" ? value.toFixed(4) : value;
}

function renderHeroStats() {
  const overview = state.overview;
  const metrics = overview.metrics.multi_platform || {};
  const mean = metrics.semantic_hybrid_gbdt_mean_multi_accuracy || metrics.lexical_modern_gbdt_mean_multi_accuracy || 0;
  elements.heroStats.innerHTML = `
    <div>
      <p class="eyebrow">Snapshot</p>
      <h2>Current run</h2>
      <p class="muted">Semantic device: ${overview.semantic_device}. Loaded models: ${overview.models.length}.</p>
    </div>
    <div class="stat-grid">
      <div class="stat-card"><div class="stat-label">Profiles</div><div class="stat-value">${Object.values(overview.dataset_counts).reduce((a, b) => a + b, 0)}</div></div>
      <div class="stat-card"><div class="stat-label">3-platform groups</div><div class="stat-value">${overview.identity_counts.triple_platform_groups}</div></div>
      <div class="stat-card"><div class="stat-label">Best multi mean</div><div class="stat-value">${mean.toFixed(4)}</div></div>
      <div class="stat-card"><div class="stat-label">Semantic cache</div><div class="stat-value">Loaded</div></div>
    </div>
  `;
}

function renderTables() {
  const pairwise = state.overview.metrics.pairwise || [];
  const multi = state.overview.metrics.multi_platform || {};

  elements.pairwiseTable.innerHTML = `
    <table>
      <thead>
        <tr>
          <th>Task</th>
          <th>RF</th>
          <th>Lexical</th>
          <th>Hybrid</th>
        </tr>
      </thead>
      <tbody>
        ${pairwise.map((row) => `
          <tr>
            <td>${row.source_platform} → ${row.target_platform}</td>
            <td>${formatNumber(row.metrics.linksocial_rf_accuracy)}</td>
            <td>${formatNumber(row.metrics.lexical_modern_gbdt_accuracy)}</td>
            <td>${formatNumber(row.metrics.semantic_hybrid_gbdt_accuracy)}</td>
          </tr>
        `).join("")}
      </tbody>
    </table>
  `;

  const multiRows = [
    ["baseline", multi.baseline_mean_multi_accuracy],
    ["semantic_cosine", multi.semantic_cosine_mean_multi_accuracy],
    ["linksocial_rf", multi.linksocial_rf_mean_multi_accuracy],
    ["lexical_modern_gbdt", multi.lexical_modern_gbdt_mean_multi_accuracy],
    ["semantic_hybrid_gbdt", multi.semantic_hybrid_gbdt_mean_multi_accuracy],
  ];
  elements.multiTable.innerHTML = `
    <table>
      <thead><tr><th>Model</th><th>Mean accuracy</th></tr></thead>
      <tbody>
        ${multiRows.map(([name, value]) => `<tr><td>${name}</td><td>${formatNumber(value)}</td></tr>`).join("")}
      </tbody>
    </table>
  `;
}

function syncModelOptions() {
  elements.modelSelect.innerHTML = state.overview.models
    .map((model) => `<option value="${model}" ${model === state.selectedModel ? "selected" : ""}>${model}</option>`)
    .join("");
}

function renderProfiles(profiles) {
  elements.profileList.innerHTML = profiles.map((profile) => `
    <div class="profile-item ${state.selectedProfile?.profile_id === profile.profile_id ? "active" : ""}" data-profile-id="${profile.profile_id}">
      <div>${badge(profile.platform)}</div>
      <h3>${profile.username || profile.full_name || profile.identity_id}</h3>
      <div class="profile-meta muted">${profile.full_name || "No full name"} · ${profile.platform_count} platform(s)</div>
      <p class="muted">${(profile.bio || "No bio").slice(0, 120)}</p>
    </div>
  `).join("");

  document.querySelectorAll(".profile-item").forEach((node) => {
    node.addEventListener("click", async () => {
      const profileId = node.dataset.profileId;
      const detail = await api(`/api/profiles/${encodeURIComponent(profileId)}`);
      state.selectedProfile = detail.profile;
      if (state.selectedProfile.platform === state.selectedTarget) {
        const fallback = ["google_plus", "instagram", "twitter"].find((platform) => platform !== state.selectedProfile.platform);
        state.selectedTarget = fallback;
        elements.targetPlatform.value = fallback;
      }
      renderInspectorIntro(detail);
      await runLinkage();
    });
  });
}

function renderInspectorIntro(detail) {
  const linked = detail.linked_profiles || [];
  elements.inspector.innerHTML = `
    <div class="inspector-grid">
      <div>${badge(detail.profile.platform)}</div>
      <h3>${detail.profile.username || detail.profile.full_name || detail.profile.identity_id}</h3>
      <p class="muted">${detail.profile.bio || "No bio available."}</p>
      <div class="score-row">
        ${linked.map((profile) => `<span class="model-chip">${platformLabel(profile.platform)} truth</span>`).join("")}
      </div>
    </div>
  `;
}

function renderGraph(response) {
  const width = elements.graphSvg.clientWidth || 860;
  const height = elements.graphSvg.clientHeight || 540;
  elements.graphSvg.setAttribute("viewBox", `0 0 ${width} ${height}`);
  const source = response.graph.nodes.find((node) => node.kind === "source");
  const candidates = response.graph.nodes.filter((node) => node.kind === "candidate");
  const centerX = width / 2;
  const centerY = height / 2;
  const radius = Math.min(width, height) * 0.32;

  const positioned = candidates.map((node, index) => {
    const angle = (Math.PI * 2 * index) / Math.max(1, candidates.length) - Math.PI / 2;
    return {
      ...node,
      x: centerX + Math.cos(angle) * radius,
      y: centerY + Math.sin(angle) * radius,
    };
  });

  const lines = positioned.map((node) => {
    const edge = response.graph.edges.find((item) => item.target === node.id);
    return `<line x1="${centerX}" y1="${centerY}" x2="${node.x}" y2="${node.y}" stroke="${edge.is_truth ? "#f5f7fb" : "#b8d8ff"}" stroke-width="${1.5 + edge.score * 4.5}" stroke-opacity="${edge.is_truth ? 0.8 : 0.18 + edge.score * 0.42}" />`;
  }).join("");

  const sourceNode = `
    <circle cx="${centerX}" cy="${centerY}" r="54" fill="#121821" stroke="#b8d8ff" stroke-opacity="0.5" stroke-width="1.5" />
    <text x="${centerX}" y="${centerY - 6}" text-anchor="middle" fill="#f5f7fb" font-size="15" font-weight="700">${(source.label || "source").slice(0, 14)}</text>
    <text x="${centerX}" y="${centerY + 18}" text-anchor="middle" fill="#909aa7" font-size="11">${platformLabel(source.platform)}</text>
  `;

  const candidateNodes = positioned.map((node) => `
    <g class="graph-node" data-node-id="${node.id}">
      <circle cx="${node.x}" cy="${node.y}" r="${22 + node.score * 13}" fill="rgba(184,216,255,0.11)" stroke="#b8d8ff" stroke-opacity="0.45" stroke-width="1.2" />
      <text x="${node.x}" y="${node.y - 2}" text-anchor="middle" fill="#f5f7fb" font-size="11" font-weight="700">${(node.label || "candidate").slice(0, 12)}</text>
      <text x="${node.x}" y="${node.y + 14}" text-anchor="middle" fill="#909aa7" font-size="10">${node.score.toFixed(2)}</text>
    </g>
  `).join("");

  elements.graphSvg.innerHTML = `${lines}${sourceNode}${candidateNodes}`;
  document.querySelectorAll(".graph-node").forEach((node) => {
    node.addEventListener("click", () => {
      state.selectedCandidateId = node.dataset.nodeId;
      renderCandidateInspector();
    });
  });
}

function renderCandidateInspector() {
  const response = state.lastLinkResponse;
  if (!response) {
    return;
  }
  const active = response.candidates.find((candidate) => candidate.profile.profile_id === state.selectedCandidateId) || response.candidates[0];
  if (!active) {
    return;
  }
  state.selectedCandidateId = active.profile.profile_id;
  const featureRows = Object.entries(active.features)
    .sort((left, right) => right[1] - left[1])
    .slice(0, 8)
    .map(([name, value]) => `<div class="score-chip">${name}: ${Number(value).toFixed(4)}</div>`)
    .join("");

  const modelRows = Object.entries(active.scores)
    .sort((left, right) => right[1] - left[1])
    .map(([name, value]) => `<div class="score-chip">${name}: ${Number(value).toFixed(4)}</div>`)
    .join("");

  elements.inspector.innerHTML = `
    <div class="candidate-card active">
      <div>${badge(active.profile.platform)}</div>
      <h3>${active.profile.username || active.profile.full_name || active.profile.identity_id}</h3>
      <div class="candidate-meta muted">${active.is_truth ? "Ground-truth linked identity" : "Candidate profile"}</div>
      <p class="muted">${active.profile.bio || "No bio available."}</p>
      <div class="score-row">${active.won_models.map((name) => `<span class="model-chip">${name} winner</span>`).join("")}</div>
      <h4>Model scores</h4>
      <div class="score-row">${modelRows}</div>
      <h4>Top evidence</h4>
      <div class="score-row">${featureRows}</div>
    </div>
  `;
}

async function runLinkage() {
  if (!state.selectedProfile) {
    return;
  }
  if (state.selectedProfile.platform === state.selectedTarget) {
    elements.inspector.innerHTML = `<p class="muted">Choose a target platform different from the source profile platform.</p>`;
    return;
  }
  elements.inspector.innerHTML = `<p class="muted">Scoring candidate profiles with ${state.selectedModel}...</p>`;
  const response = await api(
    `/api/link/${encodeURIComponent(state.selectedProfile.profile_id)}?target_platform=${encodeURIComponent(state.selectedTarget)}&model=${encodeURIComponent(state.selectedModel)}&top_k=8`
  );
  state.lastLinkResponse = response;
  state.selectedCandidateId = response.candidates[0]?.profile.profile_id || null;
  renderGraph(response);
  renderCandidateInspector();
}

async function searchProfiles() {
  const platform = elements.platformFilter.value;
  const query = elements.profileQuery.value;
  const params = new URLSearchParams();
  if (platform) {
    params.set("platform", platform);
  }
  if (query) {
    params.set("q", query);
  }
  params.set("limit", "30");
  const profiles = await api(`/api/profiles?${params.toString()}`);
  renderProfiles(profiles);
}

async function bootstrap() {
  state.overview = await api("/api/overview");
  renderHeroStats();
  renderTables();
  syncModelOptions();
  await searchProfiles();
}

elements.searchButton.addEventListener("click", searchProfiles);
elements.linkButton.addEventListener("click", runLinkage);
elements.targetPlatform.addEventListener("change", (event) => {
  state.selectedTarget = event.target.value;
});
elements.modelSelect.addEventListener("change", (event) => {
  state.selectedModel = event.target.value;
});

bootstrap().catch((error) => {
  elements.inspector.innerHTML = `<p class="muted">${error.message}</p>`;
});
