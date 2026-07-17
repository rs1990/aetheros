/**
 * Phase 1+2 — Route planning screen.
 * Input: origin (from GPS or text) + destination.
 * Output: 2–3 scored route options with landmark counts and crowd scores.
 */

import React, { useState } from 'react';
import {
  View, Text, TextInput, StyleSheet, TouchableOpacity,
  ScrollView, ActivityIndicator,
} from 'react-native';
import MapView, { Polyline, Marker } from 'react-native-maps';
import { api } from '../services/api';

export function RouteScreen({ route: navRoute, navigation }) {
  const origin = navRoute.params?.origin;

  const [destination, setDestination] = useState('');
  const [routes, setRoutes]           = useState([]);
  const [loading, setLoading]         = useState(false);
  const [error, setError]             = useState(null);
  const [selected, setSelected]       = useState(0);

  // Stub: parse "lat,lon" from destination input or use geocoded result
  function parseCoords(str) {
    const parts = str.split(',').map(Number);
    if (parts.length === 2 && !isNaN(parts[0]) && !isNaN(parts[1])) {
      return { lat: parts[0], lon: parts[1] };
    }
    return null;
  }

  async function handleScore() {
    const dest = parseCoords(destination);
    if (!dest) {
      setError('Enter destination as "lat,lon" (e.g. 37.8199,-122.4783 for Golden Gate)');
      return;
    }
    if (!origin) {
      setError('Location not available yet');
      return;
    }

    setLoading(true);
    setError(null);
    setRoutes([]);

    try {
      const { routes: scored } = await api.routes.score(
        origin,
        dest,
        new Date().toISOString()
      );
      setRoutes(scored);
      setSelected(0);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }

  const activeRoute = routes[selected];

  return (
    <View style={styles.container}>
      {/* Map preview */}
      <MapView
        style={styles.map}
        initialRegion={{
          latitude: origin?.lat ?? 37.7749,
          longitude: origin?.lon ?? -122.4194,
          latitudeDelta: 0.15,
          longitudeDelta: 0.15,
        }}
      >
        {origin && (
          <Marker
            coordinate={{ latitude: origin.lat, longitude: origin.lon }}
            title="You are here"
            pinColor="#2563EB"
          />
        )}

        {activeRoute && (
          <>
            <Polyline
              coordinates={activeRoute.points.map(([lat, lon]) => ({ latitude: lat, longitude: lon }))}
              strokeColor="#2563EB"
              strokeWidth={3}
            />
            {activeRoute.landmarks.map((lm) => (
              <Marker
                key={lm.id}
                coordinate={{ latitude: lm.latitude, longitude: lm.longitude }}
                title={lm.name}
                pinColor="#F59E0B"
              />
            ))}
          </>
        )}
      </MapView>

      {/* Bottom sheet */}
      <View style={styles.sheet}>
        <View style={styles.inputRow}>
          <TextInput
            style={styles.input}
            placeholder='Destination (lat,lon or name)'
            value={destination}
            onChangeText={setDestination}
            returnKeyType="go"
            onSubmitEditing={handleScore}
          />
          <TouchableOpacity style={styles.goBtn} onPress={handleScore} disabled={loading}>
            {loading
              ? <ActivityIndicator color="#fff" size="small" />
              : <Text style={styles.goBtnText}>Go</Text>
            }
          </TouchableOpacity>
        </View>

        {error ? <Text style={styles.error}>{error}</Text> : null}

        {routes.length > 0 && (
          <ScrollView horizontal showsHorizontalScrollIndicator={false} style={styles.routeList}>
            {routes.map((r, idx) => (
              <TouchableOpacity
                key={r.id}
                style={[styles.routeCard, idx === selected && styles.routeCardActive]}
                onPress={() => setSelected(idx)}
              >
                <Text style={[styles.routeLabel, idx === selected && styles.routeLabelActive]}>
                  {r.label}
                </Text>
                <Text style={styles.routeTime}>⏱ {r.duration_minutes} min</Text>
                <Text style={styles.routeLandmarks}>🏛 {r.landmark_count} landmarks</Text>
                <Text style={styles.routeScore}>Score: {r.score}</Text>
                {r.crowd_penalty > 10 && (
                  <Text style={styles.crowdWarning}>⚠ Crowds expected</Text>
                )}
              </TouchableOpacity>
            ))}
          </ScrollView>
        )}

        {activeRoute && (
          <TouchableOpacity
            style={styles.startBtn}
            onPress={() => navigation.navigate('Tour', { route: activeRoute, origin })}
          >
            <Text style={styles.startBtnText}>Start Tour →</Text>
          </TouchableOpacity>
        )}
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container:  { flex: 1 },
  map:        { flex: 1 },
  sheet: {
    backgroundColor: '#fff', borderTopLeftRadius: 20, borderTopRightRadius: 20,
    padding: 16, paddingBottom: 30,
    shadowColor: '#000', shadowOpacity: 0.15, shadowRadius: 10, elevation: 10,
    maxHeight: '45%',
  },
  inputRow: { flexDirection: 'row', gap: 10, marginBottom: 10 },
  input: {
    flex: 1, borderWidth: 1, borderColor: '#D1D5DB', borderRadius: 10,
    paddingHorizontal: 14, paddingVertical: 10, fontSize: 15, color: '#111827',
  },
  goBtn: { backgroundColor: '#2563EB', borderRadius: 10, paddingHorizontal: 20, justifyContent: 'center' },
  goBtnText: { color: '#fff', fontWeight: '700', fontSize: 16 },
  error: { color: '#DC2626', fontSize: 13, marginBottom: 8 },

  routeList: { marginBottom: 12 },
  routeCard: {
    borderWidth: 1.5, borderColor: '#E5E7EB', borderRadius: 12,
    padding: 12, marginRight: 10, minWidth: 140,
  },
  routeCardActive: { borderColor: '#2563EB', backgroundColor: '#EFF6FF' },
  routeLabel:  { fontWeight: '700', fontSize: 13, color: '#374151', marginBottom: 4 },
  routeLabelActive: { color: '#2563EB' },
  routeTime:   { fontSize: 12, color: '#6B7280', marginBottom: 2 },
  routeLandmarks: { fontSize: 12, color: '#6B7280', marginBottom: 2 },
  routeScore:  { fontSize: 12, fontWeight: '600', color: '#059669' },
  crowdWarning: { fontSize: 11, color: '#D97706', marginTop: 4 },

  startBtn: {
    backgroundColor: '#2563EB', borderRadius: 12, padding: 15, alignItems: 'center',
  },
  startBtnText: { color: '#fff', fontWeight: '700', fontSize: 16 },
});
