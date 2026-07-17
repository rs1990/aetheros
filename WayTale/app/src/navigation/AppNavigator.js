import React from 'react';
import { NavigationContainer } from '@react-navigation/native';
import { createNativeStackNavigator } from '@react-navigation/native-stack';
import { MapScreen } from '../screens/MapScreen';
import { RouteScreen } from '../screens/RouteScreen';
import { LandmarkDetailScreen } from '../screens/LandmarkDetailScreen';
import { TourScreen } from '../screens/TourScreen';

const Stack = createNativeStackNavigator();

export function AppNavigator() {
  return (
    <NavigationContainer>
      <Stack.Navigator
        initialRouteName="Map"
        screenOptions={{
          headerStyle: { backgroundColor: '#1E293B' },
          headerTintColor: '#F1F5F9',
          headerTitleStyle: { fontWeight: '700' },
        }}
      >
        <Stack.Screen
          name="Map"
          component={MapScreen}
          options={{ headerShown: false }}
        />
        <Stack.Screen
          name="Route"
          component={RouteScreen}
          options={{ title: 'Plan Route' }}
        />
        <Stack.Screen
          name="LandmarkDetail"
          component={LandmarkDetailScreen}
          options={({ route }) => ({ title: route.params?.landmark?.name ?? 'Landmark' })}
        />
        <Stack.Screen
          name="Tour"
          component={TourScreen}
          options={{ title: 'Live Tour', headerLeft: () => null }}
        />
      </Stack.Navigator>
    </NavigationContainer>
  );
}
