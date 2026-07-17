/**
 * Phase 3 — Live audio tour screen.
 * Watches GPS position, triggers ambient narration when entering a landmark geofence.
 * Auto-pauses when stationary for >3s mid-sentence.
 */

import React, { useState, useEffect, useRef } from 'react';
import {
  View, Text, StyleSheet, TouchableOpacity, ScrollView,
} from 'react-native';
import MapView, { Marker, Polyline, Circle } from 'react-native-maps';
import { watchLocation, isWithinRadius } from '../services/location';
import { playAudio, pauseAudio, resumeAudio, stopAudio } from '../services/audio';
import { api } from '../services/api';

const TRIGGER_RADIUS_M = 250;

export function TourScreen({ route: navRoute, navigation }) {
  const { route, origin } = navRoute.params;

  const [location, setLocation]       = useState(origin);
  const [activeLandmark, setActiveLandmark] = useState(null);
  const [triggered, setTriggered]     = useState(new Set());
  const [playing, setPlaying]         = useState(false);
  const [log, setLog]                 = useState([]);
  const cleanupRef = useRef(null);
  const stationaryTimer = useRef(null);
  const lastPosRef = useRef(null);

  useEffect(() => {
    (async () => {
      cleanupRef.current = await watchLocation(handleLocationUpdate, console.error);
    })();
    return () => {
      cleanupRef.current?.();
      stopAudio();
      clearTimeout(stationaryTimer.current);
    };
  }, []);

  async function handleLocationUpdate(loc) {
    setLocation(loc);

    // Stationary detection — pause if not moved >10m in 5s
    if (lastPosRef.current) {
      const moved = isWithinRadius(loc.lat, loc.lon, lastPosRef.current.lat, lastPosRef.current.lon, 10);
      if (moved) {
        clearTimeout(stationaryTimer.current);
        stationaryTimer.current = setTimeout(() => pauseAudio(), 5000);
      } else {
        clearTimeout(stationaryTimer.current);
        resumeAudio();
      }
    }
    lastPosRef.current = loc;

    // Geofence check against all landmarks on this route
    for (const lm of route.landmarks) {
      if (triggered.has(lm.id)) continue;
      if (isWithinRadius(loc.lat, loc.lon, lm.latitude, lm.longitude, TRIGGER_RADIUS_M)) {
        await triggerLandmark(lm);
        break;
      }
    }
  }

  async function triggerLandmark(landmark) {
    setTriggered(prev => new Set([...prev, landmark.id]));
    setActiveLandmark(landmark);
    addLog(`Approaching: ${landmark.name}`);

    try {
      const { content } = await api.landmarks.content(landmark.id, { type: 'ambient' });
      const ambient = content[0];
      if (ambient?.audio_url) {
        await playAudio(ambient.audio_url);
        setPlaying(true);
      } else if (ambient?.script) {
        addLog(`No audio — text only: ${ambient.script.slice(0, 80)}...`);
      }
    } catch (e) {
      addLog(`⚠ Could not load audio for ${landmark.name}`);
    }
  }

  function addLog(msg) {
    setLog(prev => [`${new Date().toLocaleTimeString()} — ${msg}`, ...prev.slice(0, 9)]);
  }

  async function handleDeepDive() {
    if (!activeLandmark) return;
    navigation.navigate('LandmarkDetail', { landmark: activeLandmark });
  }

  return (
    <View style={styles.container}>
      <MapView
        style={styles.map}
        showsUserLocation
        region={location ? {
          latitude: location.lat,
          longitude: location.lon,
          latitudeDelta: 0.04,
          longitudeDelta: 0.04,
        } : undefined}
      >
        <Polyline
          coordinates={route.points.map(([lat, lon]) => ({ latitude: lat, longitude: lon }))}
          strokeColor="#2563EB"
          strokeWidth={3}
        />
        {route.landmarks.map((lm) => (
          <React.Fragment key={lm.id}>
            <Marker
              coordinate={{ latitude: lm.latitude, longitude: lm.longitude }}
              title={lm.name}
              pinColor={triggered.has(lm.id) ? '#9CA3AF' : '#F59E0B'}
            />
            <Circle
              center={{ latitude: lm.latitude, longitude: lm.longitude }}
              radius={TRIGGER_RADIUS_M}
              fillColor={triggered.has(lm.id) ? 'rgba(156,163,175,0.1)' : 'rgba(245,158,11,0.1)'}
              strokeColor={triggered.has(lm.id) ? 'rgba(156,163,175,0.3)' : 'rgba(245,158,11,0.4)'}
              strokeWidth={1}
            />
          </React.Fragment>
        ))}
      </MapView>

      {/* Bottom HUD */}
      <View style={styles.hud}>
        {activeLandmark ? (
          <>
            <View style={styles.nowPlaying}>
              <Text style={styles.nowLabel}>NOW PLAYING</Text>
              <Text style={styles.nowName}>{activeLandmark.name}</Text>
              <View style={styles.controls}>
                <TouchableOpacity
                  style={styles.controlBtn}
                  onPress={() => playing ? pauseAudio().then(() => setPlaying(false)) : resumeAudio().then(() => setPlaying(true))}
                >
                  <Text style={styles.controlIcon}>{playing ? '⏸' : '▶'}</Text>
                </TouchableOpacity>
                <TouchableOpacity style={styles.deepDiveBtn} onPress={handleDeepDive}>
                  <Text style={styles.deepDiveText}>Tell me more →</Text>
                </TouchableOpacity>
              </View>
            </View>
          </>
        ) : (
          <Text style={styles.walkingText}>🚶 Walking… narration will play when you approach a landmark</Text>
        )}

        <View style={styles.logBox}>
          {log.map((entry, i) => (
            <Text key={i} style={styles.logEntry}>{entry}</Text>
          ))}
        </View>

        <TouchableOpacity style={styles.endBtn} onPress={() => { stopAudio(); navigation.goBack(); }}>
          <Text style={styles.endBtnText}>End Tour</Text>
        </TouchableOpacity>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1 },
  map:       { flex: 1 },
  hud: {
    backgroundColor: '#1E293B', padding: 16, paddingBottom: 30,
    borderTopLeftRadius: 20, borderTopRightRadius: 20,
    maxHeight: '40%',
  },
  walkingText: { color: '#94A3B8', textAlign: 'center', fontSize: 14, marginBottom: 10 },
  nowPlaying:  { marginBottom: 12 },
  nowLabel:    { fontSize: 10, color: '#2563EB', fontWeight: '700', letterSpacing: 1.5 },
  nowName:     { fontSize: 18, color: '#F1F5F9', fontWeight: '800', marginBottom: 8 },
  controls:    { flexDirection: 'row', alignItems: 'center', gap: 12 },
  controlBtn:  { backgroundColor: '#334155', borderRadius: 10, padding: 10 },
  controlIcon: { fontSize: 20, color: '#F1F5F9' },
  deepDiveBtn: { flex: 1, backgroundColor: '#2563EB', borderRadius: 10, padding: 10, alignItems: 'center' },
  deepDiveText:{ color: '#fff', fontWeight: '700', fontSize: 14 },

  logBox:   { maxHeight: 60, overflow: 'hidden', marginBottom: 10 },
  logEntry: { fontSize: 11, color: '#64748B', marginBottom: 2 },

  endBtn:    { backgroundColor: '#DC2626', borderRadius: 10, padding: 12, alignItems: 'center' },
  endBtnText:{ color: '#fff', fontWeight: '700' },
});
