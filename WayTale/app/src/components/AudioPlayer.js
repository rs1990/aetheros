import React, { useState, useEffect } from 'react';
import { View, Text, TouchableOpacity, StyleSheet, ActivityIndicator } from 'react-native';
import { playAudio, pauseAudio, resumeAudio, stopAudio } from '../services/audio';

export function AudioPlayer({ audioUrl, title, factType, onClose }) {
  const [state, setState] = useState('idle'); // idle | loading | playing | paused | error

  useEffect(() => {
    return () => { stopAudio(); };
  }, []);

  async function handlePlay() {
    try {
      setState('loading');
      await playAudio(audioUrl);
      setState('playing');
    } catch (e) {
      setState('error');
    }
  }

  async function handleToggle() {
    if (state === 'playing') {
      await pauseAudio();
      setState('paused');
    } else if (state === 'paused') {
      await resumeAudio();
      setState('playing');
    } else {
      handlePlay();
    }
  }

  const isLoading = state === 'loading';
  const isPlaying = state === 'playing';

  return (
    <View style={styles.container}>
      <View style={styles.header}>
        <Text style={styles.title} numberOfLines={1}>{title}</Text>
        {factType === 'legend' || factType === 'mixed' ? (
          <View style={styles.legendBadge}>
            <Text style={styles.legendText}>Contains legend</Text>
          </View>
        ) : null}
        <TouchableOpacity onPress={onClose} style={styles.close}>
          <Text style={styles.closeText}>✕</Text>
        </TouchableOpacity>
      </View>

      {state === 'error' ? (
        <Text style={styles.error}>Audio unavailable — text content still accessible</Text>
      ) : (
        <TouchableOpacity style={styles.playBtn} onPress={handleToggle} disabled={isLoading}>
          {isLoading
            ? <ActivityIndicator color="#fff" />
            : <Text style={styles.playIcon}>{isPlaying ? '⏸' : '▶'}</Text>
          }
          <Text style={styles.playLabel}>
            {isLoading ? 'Loading...' : isPlaying ? 'Pause' : 'Play narration'}
          </Text>
        </TouchableOpacity>
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    backgroundColor: '#1E293B',
    borderRadius: 14,
    padding: 14,
    margin: 12,
    shadowColor: '#000',
    shadowOpacity: 0.3,
    shadowRadius: 8,
    elevation: 8,
  },
  header:      { flexDirection: 'row', alignItems: 'center', marginBottom: 10 },
  title:       { flex: 1, color: '#F1F5F9', fontWeight: '700', fontSize: 14 },
  legendBadge: { backgroundColor: '#F59E0B', borderRadius: 6, paddingHorizontal: 6, paddingVertical: 2, marginRight: 8 },
  legendText:  { color: '#fff', fontSize: 10, fontWeight: '700' },
  close:       { padding: 4 },
  closeText:   { color: '#94A3B8', fontSize: 16 },
  playBtn:     { flexDirection: 'row', alignItems: 'center', backgroundColor: '#2563EB', borderRadius: 10, padding: 12, gap: 8 },
  playIcon:    { fontSize: 18 },
  playLabel:   { color: '#fff', fontWeight: '600', fontSize: 14 },
  error:       { color: '#F87171', fontSize: 12, textAlign: 'center' },
});
