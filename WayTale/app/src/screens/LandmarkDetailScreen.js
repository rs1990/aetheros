/**
 * Landmark detail screen.
 * Reads from Phase 0 pre-generated content cache.
 * Shows ambient narration + deep-dive tabs (history / geography / culture).
 * Source citations always visible per data integrity requirement.
 */

import React, { useState, useEffect } from 'react';
import {
  View, Text, ScrollView, StyleSheet, TouchableOpacity,
  ActivityIndicator, Linking, Image,
} from 'react-native';
import { api } from '../services/api';
import { AudioPlayer } from '../components/AudioPlayer';

const TABS = [
  { key: 'history',   label: 'History' },
  { key: 'geography', label: 'Geography' },
  { key: 'culture',   label: 'Culture' },
];

export function LandmarkDetailScreen({ route: navRoute }) {
  const { landmark } = navRoute.params;
  const [content, setContent]       = useState([]);
  const [loading, setLoading]       = useState(true);
  const [activeTab, setActiveTab]   = useState('history');
  const [activeAudio, setActiveAudio] = useState(null);

  useEffect(() => {
    api.landmarks.content(landmark.id)
      .then(({ content }) => setContent(content))
      .catch(console.error)
      .finally(() => setLoading(false));
  }, [landmark.id]);

  const ambientContent = content.find(c => c.content_type === 'ambient');
  const tabContent = content.find(c => c.content_type === 'deep_dive' && c.variant === activeTab);

  function getSources(item) {
    if (!item?.sources) return [];
    return JSON.parse(item.sources);
  }

  return (
    <ScrollView style={styles.container} contentContainerStyle={styles.inner}>
      {/* Header image */}
      {landmark.image_url ? (
        <Image source={{ uri: landmark.image_url }} style={styles.heroImage} resizeMode="cover" />
      ) : (
        <View style={styles.heroPlaceholder}>
          <Text style={styles.heroIcon}>🏛</Text>
        </View>
      )}

      <View style={styles.body}>
        <Text style={styles.name}>{landmark.name}</Text>
        {landmark.description ? (
          <Text style={styles.description}>{landmark.description}</Text>
        ) : null}

        {loading ? (
          <ActivityIndicator style={{ marginTop: 30 }} color="#2563EB" />
        ) : content.length === 0 ? (
          <View style={styles.noContent}>
            <Text style={styles.noContentText}>
              No verified audio content yet for this landmark.
            </Text>
          </View>
        ) : (
          <>
            {/* Ambient narration */}
            {ambientContent && (
              <View style={styles.section}>
                <Text style={styles.sectionLabel}>Ambient Tour</Text>
                {ambientContent.audio_url ? (
                  activeAudio === 'ambient' ? (
                    <AudioPlayer
                      audioUrl={ambientContent.audio_url}
                      title={`${landmark.name} — Overview`}
                      factType={ambientContent.fact_type}
                      onClose={() => setActiveAudio(null)}
                    />
                  ) : (
                    <TouchableOpacity
                      style={styles.playRow}
                      onPress={() => setActiveAudio('ambient')}
                    >
                      <Text style={styles.playRowIcon}>▶</Text>
                      <Text style={styles.playRowText}>Play ambient narration (60–90s)</Text>
                    </TouchableOpacity>
                  )
                ) : null}
                <Text style={styles.script}>{ambientContent.script}</Text>
                <FactTypeBadge factType={ambientContent.fact_type} />
                <Sources sources={getSources(ambientContent)} />
              </View>
            )}

            {/* Deep dive tabs */}
            <Text style={styles.sectionLabel}>Deep Dive</Text>
            <View style={styles.tabs}>
              {TABS.map(tab => (
                <TouchableOpacity
                  key={tab.key}
                  style={[styles.tab, activeTab === tab.key && styles.tabActive]}
                  onPress={() => setActiveTab(tab.key)}
                >
                  <Text style={[styles.tabText, activeTab === tab.key && styles.tabTextActive]}>
                    {tab.label}
                  </Text>
                </TouchableOpacity>
              ))}
            </View>

            {tabContent ? (
              <View style={styles.tabContent}>
                {tabContent.audio_url ? (
                  activeAudio === activeTab ? (
                    <AudioPlayer
                      audioUrl={tabContent.audio_url}
                      title={`${landmark.name} — ${activeTab}`}
                      factType={tabContent.fact_type}
                      onClose={() => setActiveAudio(null)}
                    />
                  ) : (
                    <TouchableOpacity
                      style={styles.playRow}
                      onPress={() => setActiveAudio(activeTab)}
                    >
                      <Text style={styles.playRowIcon}>▶</Text>
                      <Text style={styles.playRowText}>Play {activeTab} narration (3–5 min)</Text>
                    </TouchableOpacity>
                  )
                ) : null}
                <Text style={styles.script}>{tabContent.script}</Text>
                <FactTypeBadge factType={tabContent.fact_type} />
                <Sources sources={getSources(tabContent)} />
              </View>
            ) : (
              <Text style={styles.noTabContent}>
                No {activeTab} content available for this landmark yet.
              </Text>
            )}
          </>
        )}
      </View>
    </ScrollView>
  );
}

function FactTypeBadge({ factType }) {
  if (factType === 'verified') return null;
  return (
    <View style={styles.legendBanner}>
      <Text style={styles.legendBannerText}>
        ⚠ This content contains legend or folklore, clearly labeled as such.
      </Text>
    </View>
  );
}

function Sources({ sources }) {
  if (!sources?.length) return null;
  return (
    <View style={styles.sources}>
      <Text style={styles.sourcesLabel}>Sources</Text>
      {sources.map((s, i) => (
        <TouchableOpacity key={i} onPress={() => Linking.openURL(s.url)}>
          <Text style={styles.sourceLink}>
            [{i + 1}] {s.source} ↗
          </Text>
        </TouchableOpacity>
      ))}
    </View>
  );
}

const styles = StyleSheet.create({
  container:  { flex: 1, backgroundColor: '#F9FAFB' },
  inner:      { paddingBottom: 40 },
  heroImage:  { width: '100%', height: 220 },
  heroPlaceholder: {
    width: '100%', height: 180, backgroundColor: '#E2E8F0',
    alignItems: 'center', justifyContent: 'center',
  },
  heroIcon:   { fontSize: 64 },
  body:       { padding: 18 },
  name:       { fontSize: 26, fontWeight: '800', color: '#111827', marginBottom: 6 },
  description:{ fontSize: 15, color: '#4B5563', lineHeight: 22, marginBottom: 16 },
  section:    { marginBottom: 24 },
  sectionLabel: { fontSize: 13, fontWeight: '700', color: '#6B7280', textTransform: 'uppercase', letterSpacing: 1, marginBottom: 10, marginTop: 8 },
  script:     { fontSize: 15, color: '#374151', lineHeight: 24, marginTop: 8 },

  playRow: {
    flexDirection: 'row', alignItems: 'center', gap: 10,
    backgroundColor: '#EFF6FF', borderRadius: 10,
    padding: 12, marginBottom: 8,
  },
  playRowIcon: { fontSize: 16, color: '#2563EB' },
  playRowText: { color: '#2563EB', fontWeight: '600', fontSize: 14 },

  tabs:        { flexDirection: 'row', gap: 8, marginBottom: 12 },
  tab:         { flex: 1, padding: 10, borderRadius: 8, backgroundColor: '#E5E7EB', alignItems: 'center' },
  tabActive:   { backgroundColor: '#2563EB' },
  tabText:     { fontWeight: '600', color: '#374151' },
  tabTextActive: { color: '#fff' },
  tabContent:  { marginTop: 4 },
  noTabContent:{ color: '#9CA3AF', fontStyle: 'italic', marginTop: 8 },

  legendBanner: {
    backgroundColor: '#FFFBEB', borderRadius: 8, padding: 10,
    borderLeftWidth: 3, borderLeftColor: '#F59E0B', marginTop: 10,
  },
  legendBannerText: { color: '#92400E', fontSize: 13 },

  sources:      { marginTop: 14, paddingTop: 12, borderTopWidth: 1, borderTopColor: '#E5E7EB' },
  sourcesLabel: { fontSize: 12, fontWeight: '700', color: '#9CA3AF', marginBottom: 6 },
  sourceLink:   { color: '#2563EB', fontSize: 13, marginBottom: 4 },

  noContent: {
    backgroundColor: '#F3F4F6', borderRadius: 10,
    padding: 20, alignItems: 'center', marginTop: 20,
  },
  noContentText: { color: '#6B7280', textAlign: 'center', fontSize: 15 },
});
