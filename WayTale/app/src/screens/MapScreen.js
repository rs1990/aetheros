/**
 * Phase 1 — Main map screen.
 * GPS position + nearby landmarks from the pre-generated cache.
 * Stub map provider (react-native-maps default = Apple Maps on iOS, Google on Android).
 */

import React, { useState, useRef } from 'react';
import {
  View, Text, StyleSheet, TouchableOpacity, ActivityIndicator, Alert,
} from 'react-native';
import MapView, { Circle } from 'react-native-maps';
import { LandmarkPin } from '../components/LandmarkPin';
import { useLocation } from '../hooks/useLocation';
import { useNearbyLandmarks } from '../hooks/useNearbyLandmarks';

const GEOFENCE_RADIUS_M = 300;

export function MapScreen({ navigation }) {
  const { location, error: locError } = useLocation();
  const { landmarks, loading, error: ldmError } = useNearbyLandmarks(location, 5);
  const mapRef = useRef(null);

  // Auto-center on user location once
  const centeredRef = useRef(false);
  React.useEffect(() => {
    if (location && !centeredRef.current && mapRef.current) {
      centeredRef.current = true;
      mapRef.current.animateToRegion({
        latitude: location.lat,
        longitude: location.lon,
        latitudeDelta: 0.05,
        longitudeDelta: 0.05,
      }, 600);
    }
  }, [location]);

  function onLandmarkPress(landmark) {
    navigation.navigate('LandmarkDetail', { landmark });
  }

  if (locError) {
    return (
      <View style={styles.center}>
        <Text style={styles.errorText}>Location unavailable</Text>
        <Text style={styles.errorSub}>{locError}</Text>
      </View>
    );
  }

  return (
    <View style={styles.container}>
      <MapView
        ref={mapRef}
        style={styles.map}
        showsUserLocation
        showsMyLocationButton
        initialRegion={{
          latitude: 37.7749,
          longitude: -122.4194,
          latitudeDelta: 0.1,
          longitudeDelta: 0.1,
        }}
      >
        {location && (
          <Circle
            center={{ latitude: location.lat, longitude: location.lon }}
            radius={GEOFENCE_RADIUS_M}
            fillColor="rgba(37,99,235,0.1)"
            strokeColor="rgba(37,99,235,0.3)"
            strokeWidth={1}
          />
        )}

        {landmarks.map((lm) => (
          <LandmarkPin key={lm.id} landmark={lm} onPress={onLandmarkPress} />
        ))}
      </MapView>

      {/* Top bar */}
      <View style={styles.topBar}>
        <Text style={styles.appName}>WayTale</Text>
        <TouchableOpacity
          style={styles.routeBtn}
          onPress={() => navigation.navigate('Route', { origin: location })}
        >
          <Text style={styles.routeBtnText}>Plan Route →</Text>
        </TouchableOpacity>
      </View>

      {/* Loading overlay */}
      {loading && (
        <View style={styles.loadingChip}>
          <ActivityIndicator size="small" color="#2563EB" />
          <Text style={styles.loadingText}>Loading landmarks...</Text>
        </View>
      )}

      {/* Landmark count chip */}
      {!loading && landmarks.length > 0 && (
        <View style={styles.countChip}>
          <Text style={styles.countText}>🏛 {landmarks.length} landmarks nearby</Text>
        </View>
      )}

      {ldmError && (
        <View style={styles.errorChip}>
          <Text style={styles.errorChipText}>⚠ {ldmError}</Text>
        </View>
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  container:     { flex: 1 },
  map:           { flex: 1 },
  center:        { flex: 1, alignItems: 'center', justifyContent: 'center', padding: 24 },
  errorText:     { fontSize: 18, fontWeight: '700', color: '#DC2626' },
  errorSub:      { color: '#6B7280', marginTop: 6, textAlign: 'center' },

  topBar: {
    position: 'absolute', top: 50, left: 12, right: 12,
    flexDirection: 'row', alignItems: 'center',
    backgroundColor: '#fff', borderRadius: 14, padding: 12,
    shadowColor: '#000', shadowOpacity: 0.15, shadowRadius: 8, elevation: 5,
  },
  appName:    { flex: 1, fontWeight: '800', fontSize: 20, color: '#111827', letterSpacing: 0.5 },
  routeBtn:   { backgroundColor: '#2563EB', borderRadius: 8, paddingHorizontal: 14, paddingVertical: 8 },
  routeBtnText: { color: '#fff', fontWeight: '700', fontSize: 13 },

  loadingChip: {
    position: 'absolute', bottom: 30, alignSelf: 'center',
    flexDirection: 'row', alignItems: 'center', gap: 8,
    backgroundColor: '#fff', borderRadius: 20, paddingHorizontal: 16, paddingVertical: 10,
    shadowColor: '#000', shadowOpacity: 0.15, shadowRadius: 6, elevation: 4,
  },
  loadingText: { color: '#374151', fontWeight: '500' },
  countChip: {
    position: 'absolute', bottom: 30, alignSelf: 'center',
    backgroundColor: '#fff', borderRadius: 20, paddingHorizontal: 16, paddingVertical: 10,
    shadowColor: '#000', shadowOpacity: 0.12, shadowRadius: 6, elevation: 4,
  },
  countText: { color: '#111827', fontWeight: '600', fontSize: 13 },
  errorChip: {
    position: 'absolute', bottom: 30, alignSelf: 'center',
    backgroundColor: '#FEF2F2', borderRadius: 20, paddingHorizontal: 16, paddingVertical: 10,
  },
  errorChipText: { color: '#DC2626', fontSize: 12 },
});
