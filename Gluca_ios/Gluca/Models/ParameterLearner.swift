import Foundation

struct SleepSessionSignal: Codable, Equatable {
    let start: Date
    let end: Date
    let totalHours: Double
    let deepHours: Double?
    let remHours: Double?
    let awakeningCount: Int?
    let qualityScore: Double?
    let source: String?
}

struct WorkoutSignalSummary: Codable, Equatable {
    let activityType: String
    let intensity: String?
    let durationMinutes: Double
    let start: Date
    let end: Date?
    let source: String?
    let energyKilocalories: Double?
    let distanceMeters: Double?
    let isIndoor: Bool?
    let brandName: String?
    let whoopStrain: Double?
    let avgHeartRateBpm: Double?
    let maxHeartRateBpm: Double?

    init(
        activityType: String,
        intensity: String? = nil,
        durationMinutes: Double,
        start: Date,
        end: Date? = nil,
        source: String? = nil,
        energyKilocalories: Double? = nil,
        distanceMeters: Double? = nil,
        isIndoor: Bool? = nil,
        brandName: String? = nil,
        whoopStrain: Double? = nil,
        avgHeartRateBpm: Double? = nil,
        maxHeartRateBpm: Double? = nil
    ) {
        self.activityType = activityType
        self.intensity = intensity
        self.durationMinutes = durationMinutes
        self.start = start
        self.end = end
        self.source = source
        self.energyKilocalories = energyKilocalories
        self.distanceMeters = distanceMeters
        self.isIndoor = isIndoor
        self.brandName = brandName
        self.whoopStrain = whoopStrain
        self.avgHeartRateBpm = avgHeartRateBpm
        self.maxHeartRateBpm = maxHeartRateBpm
    }
}

struct ContextSignalSummary: Codable, Equatable {
    let sleepSessionsLast7d: Int
    let averageSleepHoursLast7d: Double?
    let latestSleepHours: Double?
    let latestSleepSession: SleepSessionSignal?
    let latestRestingHeartRateBpm: Double?
    let latestHRVMs: Double?
    let stepsLast24h: Double
    let stepLoadLast6h: Double
    let recentWorkoutIntensity: String?
    let latestWorkout: WorkoutSignalSummary?
    let recentWorkouts: [WorkoutSignalSummary]
    let workoutCountLast24h: Int
    let workoutMinutesLast24h: Double
    let workoutCountLast7d: Int
    let workoutMinutesLast7d: Double
    let recentWorkoutTypes: [String]
    let recentStressFlags: Int
    let recentIllnessFlags: Int

    init(
        sleepSessionsLast7d: Int = 0,
        averageSleepHoursLast7d: Double? = nil,
        latestSleepHours: Double? = nil,
        latestSleepSession: SleepSessionSignal? = nil,
        latestRestingHeartRateBpm: Double? = nil,
        latestHRVMs: Double? = nil,
        stepsLast24h: Double = 0,
        stepLoadLast6h: Double = 0,
        recentWorkoutIntensity: String? = nil,
        latestWorkout: WorkoutSignalSummary? = nil,
        recentWorkouts: [WorkoutSignalSummary] = [],
        workoutCountLast24h: Int = 0,
        workoutMinutesLast24h: Double = 0,
        workoutCountLast7d: Int = 0,
        workoutMinutesLast7d: Double = 0,
        recentWorkoutTypes: [String] = [],
        recentStressFlags: Int = 0,
        recentIllnessFlags: Int = 0
    ) {
        self.sleepSessionsLast7d = sleepSessionsLast7d
        self.averageSleepHoursLast7d = averageSleepHoursLast7d
        self.latestSleepHours = latestSleepHours
        self.latestSleepSession = latestSleepSession
        self.latestRestingHeartRateBpm = latestRestingHeartRateBpm
        self.latestHRVMs = latestHRVMs
        self.stepsLast24h = stepsLast24h
        self.stepLoadLast6h = stepLoadLast6h
        self.recentWorkoutIntensity = recentWorkoutIntensity
        self.latestWorkout = latestWorkout
        self.recentWorkouts = recentWorkouts
        self.workoutCountLast24h = workoutCountLast24h
        self.workoutMinutesLast24h = workoutMinutesLast24h
        self.workoutCountLast7d = workoutCountLast7d
        self.workoutMinutesLast7d = workoutMinutesLast7d
        self.recentWorkoutTypes = recentWorkoutTypes
        self.recentStressFlags = recentStressFlags
        self.recentIllnessFlags = recentIllnessFlags
    }
}

enum LearnerObservationKind: String, Codable {
    case correctionISF = "correction_isf"
    case carbRatio = "carb_ratio"
    case carbAbsorption = "carb_absorption"
    case exerciseBoost = "exercise_boost"
    case exerciseDecay = "exercise_decay"
    case dawnRise = "dawn_rise"
}

struct LearnerObservationSignals: Codable, Equatable {
    let activityLevel: Double
    let sleepQuality: Double
    let stressLevel: Double
    let illnessFlag: Double
    let stepsLast6h: Double
    let workoutIntensityScore: Double
    let workoutMinutesLast6h: Double?
    let workoutWhoopStrain: Double?
    let workoutType: String?
}

struct LearnerObservation: Identifiable, Codable, Equatable {
    let id: String
    let kind: LearnerObservationKind
    let anchorEventID: String
    let anchorTimestamp: Date
    let resolvedAt: Date
    let observedValue: Double
    let secondaryValue: Double?
    let timeBucket: String?
    let confidence: InsightConfidence
    let signals: LearnerObservationSignals
}

struct LearnerActionOutcomeRecord: Identifiable, Codable, Equatable {
    let id: String
    let kind: LearnerObservationKind
    let anchorEventID: String
    let anchorTimestamp: Date
    let resolvedAt: Date
    let glucoseBefore: Double?
    let glucoseAfter: Double?
    let primaryInputValue: Double
    let secondaryInputValue: Double?
    let derivedOutcomeValue: Double
    let timeBucket: String?
    let confidence: InsightConfidence
    let signals: LearnerObservationSignals
}

struct LocalParameterSnapshot: Identifiable, Codable {
    let id: String
    let snapshotDate: Date
    let recordedAt: Date
    let parameterOutput: BayesianParameterOutput
    let syncedAt: Date?

    func markSynced(at date: Date = .now) -> LocalParameterSnapshot {
        LocalParameterSnapshot(
            id: id,
            snapshotDate: snapshotDate,
            recordedAt: recordedAt,
            parameterOutput: parameterOutput,
            syncedAt: date
        )
    }
}

struct PersistedParameterLearnerState: Codable {
    let schemaVersion: Int
    let updatedAt: Date
    let parameterOutput: BayesianParameterOutput?
    let contextSignals: ContextSignalSummary
    let records: [LearnerActionOutcomeRecord]
    let observations: [LearnerObservation]
    let snapshots: [LocalParameterSnapshot]

    private enum CodingKeys: String, CodingKey {
        case schemaVersion
        case updatedAt
        case parameterOutput
        case contextSignals
        case records
        case observations
        case snapshots
    }

    init(
        schemaVersion: Int,
        updatedAt: Date,
        parameterOutput: BayesianParameterOutput?,
        contextSignals: ContextSignalSummary,
        records: [LearnerActionOutcomeRecord],
        observations: [LearnerObservation],
        snapshots: [LocalParameterSnapshot]
    ) {
        self.schemaVersion = schemaVersion
        self.updatedAt = updatedAt
        self.parameterOutput = parameterOutput
        self.contextSignals = contextSignals
        self.records = records
        self.observations = observations
        self.snapshots = snapshots
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        schemaVersion = try container.decodeIfPresent(Int.self, forKey: .schemaVersion) ?? 1
        updatedAt = try container.decodeIfPresent(Date.self, forKey: .updatedAt) ?? .distantPast
        parameterOutput = try container.decodeIfPresent(BayesianParameterOutput.self, forKey: .parameterOutput)
        contextSignals = try container.decodeIfPresent(ContextSignalSummary.self, forKey: .contextSignals) ?? ContextSignalSummary()
        records = try container.decodeIfPresent([LearnerActionOutcomeRecord].self, forKey: .records) ?? []
        observations = try container.decodeIfPresent([LearnerObservation].self, forKey: .observations) ?? []
        snapshots = try container.decodeIfPresent([LocalParameterSnapshot].self, forKey: .snapshots) ?? []
    }

    func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(schemaVersion, forKey: .schemaVersion)
        try container.encode(updatedAt, forKey: .updatedAt)
        try container.encodeIfPresent(parameterOutput, forKey: .parameterOutput)
        try container.encode(contextSignals, forKey: .contextSignals)
        try container.encode(records, forKey: .records)
        try container.encode(observations, forKey: .observations)
        try container.encode(snapshots, forKey: .snapshots)
    }
}

enum ParameterLearnerStore {
    private static let defaultsKey = "com.railabs.gluca.parameter.learner"

    static func loadPersisted() -> PersistedParameterLearnerState {
        guard let data = UserDefaults.standard.data(forKey: defaultsKey),
              let decoded = try? JSONDecoder().decode(PersistedParameterLearnerState.self, from: data)
        else {
            return PersistedParameterLearnerState(
                schemaVersion: 2,
                updatedAt: .distantPast,
                parameterOutput: nil,
                contextSignals: ContextSignalSummary(
                    sleepSessionsLast7d: 0,
                    averageSleepHoursLast7d: nil,
                    latestSleepHours: nil,
                    latestRestingHeartRateBpm: nil,
                    latestHRVMs: nil,
                    stepsLast24h: 0,
                    stepLoadLast6h: 0,
                    recentWorkoutIntensity: nil,
                    recentWorkouts: [],
                    workoutCountLast24h: 0,
                    workoutMinutesLast24h: 0,
                    recentStressFlags: 0,
                    recentIllnessFlags: 0
                ),
                records: [],
                observations: [],
                snapshots: []
            )
        }
        return decoded
    }

    static func persist(_ state: PersistedParameterLearnerState) {
        guard let encoded = try? JSONEncoder().encode(state) else { return }
        UserDefaults.standard.set(encoded, forKey: defaultsKey)
    }

    static func clear() {
        UserDefaults.standard.removeObject(forKey: defaultsKey)
    }
}

enum ParameterLearner {
    private static let observationRetentionDays = 90
    private static let maxSnapshots = 120

    static func reconcile(
        existing: PersistedParameterLearnerState,
        timeline: [GlucoseTimelineEvent],
        inferredMealEvents: [InferredMealEvent] = [],
        readings: [GlucoseReading],
        therapyProfile: TherapyProfileContext?,
        feedQuality: InsightConfidence,
        cohort: String,
        clinicalSex: String = ClinicalSex.unspecified.rawValue,
        weightKg: Double? = nil,
        historicalSummary: HistoricalInsightWindowSummary?,
        statisticalAnalysis: StatisticalAnalysis? = nil,
        referenceTime: Date = .now
    ) -> PersistedParameterLearnerState {
        let sortedTimeline = timeline.sorted { $0.timestamp < $1.timestamp }
        let sortedReadings = readings.sorted { $0.timestamp < $1.timestamp }
        let availableUntil = min(referenceTime, sortedReadings.last?.timestamp ?? referenceTime)
        let contextSignals = buildContextSignalSummary(from: sortedTimeline, referenceTime: referenceTime)

        let extractedRecords = extractRecords(
            timeline: sortedTimeline,
            inferredMealEvents: inferredMealEvents,
            readings: sortedReadings,
            therapyProfile: therapyProfile,
            cohort: cohort,
            clinicalSex: clinicalSex,
            weightKg: weightKg,
            availableUntil: availableUntil
        )
        let extractedRecordIDs = Set(extractedRecords.map(\.id))
        let revalidationStart: Date? = {
            guard let timelineStart = sortedTimeline.first?.timestamp,
                  let readingsStart = sortedReadings.first?.timestamp
            else {
                return nil
            }
            return max(timelineStart, readingsStart)
        }()

        var mergedRecordsByID = Dictionary(uniqueKeysWithValues: existing.records.compactMap { record -> (String, LearnerActionOutcomeRecord)? in
            if record.kind == .correctionISF,
               let revalidationStart,
               record.anchorTimestamp >= revalidationStart,
               record.resolvedAt <= availableUntil,
               !extractedRecordIDs.contains(record.id) {
                return nil
            }
            return (record.id, record)
        })
        for record in extractedRecords {
            mergedRecordsByID[record.id] = record
        }

        let retentionCutoff = Calendar.current.date(byAdding: .day, value: -observationRetentionDays, to: referenceTime) ?? .distantPast
        let mergedRecords = mergedRecordsByID.values
            .filter { $0.anchorTimestamp >= retentionCutoff }
            .sorted { lhs, rhs in
                if lhs.anchorTimestamp == rhs.anchorTimestamp {
                    return lhs.id < rhs.id
                }
                return lhs.anchorTimestamp < rhs.anchorTimestamp
            }
        let mergedObservations = deriveObservations(from: mergedRecords)

        let parameterOutput = buildParameterOutput(
            observations: mergedObservations,
            therapyProfile: therapyProfile,
            feedQuality: feedQuality,
            cohort: cohort,
            clinicalSex: clinicalSex,
            weightKg: weightKg,
            generatedAt: referenceTime,
            historicalSummary: historicalSummary,
            statisticalAnalysis: statisticalAnalysis
        )

        let snapshots = recordSnapshot(
            parameterOutput: parameterOutput,
            existing: existing.snapshots,
            referenceTime: referenceTime
        )

        return PersistedParameterLearnerState(
            schemaVersion: 2,
            updatedAt: referenceTime,
            parameterOutput: parameterOutput,
            contextSignals: contextSignals,
            records: mergedRecords,
            observations: mergedObservations,
            snapshots: snapshots
        )
    }

    static func buildContextSignalSummary(
        from timeline: [GlucoseTimelineEvent],
        referenceTime: Date = .now
    ) -> ContextSignalSummary {
        let last7d = referenceTime.addingTimeInterval(-7 * 24 * 3600)
        let last24h = referenceTime.addingTimeInterval(-24 * 3600)
        let last6h = referenceTime.addingTimeInterval(-6 * 3600)
        let sleepEvents = timeline
            .filter { $0.kind == .sleep && $0.timestamp >= last7d }
            .sorted { $0.timestamp > $1.timestamp }
        let stepEvents = timeline
            .filter { $0.kind == .steps }
        let workoutEvents = timeline
            .filter { $0.kind == .exercise && $0.timestamp >= last7d }
            .sorted { $0.timestamp > $1.timestamp }
        let contextSignalEvents = timeline
            .filter { $0.kind == .contextSignal && $0.timestamp >= referenceTime.addingTimeInterval(-48 * 3600) }
            .sorted { $0.timestamp > $1.timestamp }
        let recentNotes = timeline
            .filter { $0.kind == .note && $0.timestamp >= last7d }

        let sleepHours = sleepEvents.compactMap(\.value?.amount)
        let stepsLast24h = stepEvents
            .filter { $0.timestamp >= last24h }
            .compactMap(\.value?.amount)
            .reduce(0, +)
        let stepLoadLast6h = stepEvents
            .filter { $0.timestamp >= last6h }
            .compactMap(\.value?.amount)
            .reduce(0, +)
        let recentStressFlags = recentNotes.filter { noteHasStressFlag($0) }.count
        let recentIllnessFlags = recentNotes.filter { noteHasIllnessFlag($0) }.count
        let latestSleepSession = sleepEvents.compactMap(sleepSessionSummary(from:)).first
        let latestRestingHeartRate = contextMetricValue(
            from: contextSignalEvents,
            metricKey: "resting_heart_rate_bpm"
        )
        let latestHRV = contextMetricValue(
            from: contextSignalEvents,
            metricKey: "hrv_ms"
        )
        let workoutSummaries = workoutEvents.compactMap(workoutSummary(from:))
        let latestWorkout = workoutSummaries.first
        let recentWorkoutIntensity = latestWorkout?.intensity
            ?? timeline
                .filter { $0.kind == .exercise && $0.timestamp >= last24h }
                .sorted { $0.timestamp > $1.timestamp }
                .compactMap { parseWorkoutIntensity(from: $0.detail, metadata: $0.metadata) }
                .first
        let workoutEventsLast24h = workoutEvents.filter { $0.timestamp >= last24h }
        let workoutMinutesLast24h = workoutEventsLast24h
            .compactMap(\.value?.amount)
            .reduce(0, +)
        let workoutMinutesLast7d = workoutEvents
            .compactMap(\.value?.amount)
            .reduce(0, +)
        let recentWorkoutTypes = Array(NSOrderedSet(array: workoutEvents.compactMap {
            workoutType(from: $0)
        })) as? [String] ?? []

        return ContextSignalSummary(
            sleepSessionsLast7d: sleepEvents.count,
            averageSleepHoursLast7d: average(sleepHours),
            latestSleepHours: sleepHours.first,
            latestSleepSession: latestSleepSession,
            latestRestingHeartRateBpm: latestRestingHeartRate,
            latestHRVMs: latestHRV,
            stepsLast24h: stepsLast24h,
            stepLoadLast6h: stepLoadLast6h,
            recentWorkoutIntensity: recentWorkoutIntensity,
            latestWorkout: latestWorkout,
            recentWorkouts: Array(workoutSummaries.prefix(4)),
            workoutCountLast24h: workoutEventsLast24h.count,
            workoutMinutesLast24h: workoutMinutesLast24h,
            workoutCountLast7d: workoutEvents.count,
            workoutMinutesLast7d: workoutMinutesLast7d,
            recentWorkoutTypes: Array(recentWorkoutTypes.prefix(4)),
            recentStressFlags: recentStressFlags,
            recentIllnessFlags: recentIllnessFlags
        )
    }

    static func pendingSnapshots(from state: PersistedParameterLearnerState) -> [LocalParameterSnapshot] {
        state.snapshots.filter { $0.syncedAt == nil }
    }

    static func markSnapshotSynced(
        _ snapshotID: String,
        in state: PersistedParameterLearnerState,
        syncedAt: Date = .now
    ) -> PersistedParameterLearnerState {
        let updatedSnapshots = state.snapshots.map { snapshot in
            snapshot.id == snapshotID ? snapshot.markSynced(at: syncedAt) : snapshot
        }
        return PersistedParameterLearnerState(
            schemaVersion: state.schemaVersion,
            updatedAt: state.updatedAt,
            parameterOutput: state.parameterOutput,
            contextSignals: state.contextSignals,
            records: state.records,
            observations: state.observations,
            snapshots: updatedSnapshots
        )
    }

    static func snapshotUploadRow(
        snapshot: LocalParameterSnapshot,
        userID: String
    ) -> [String: Any] {
        let output = snapshot.parameterOutput
        return [
            "user_id": userID,
            "snapshot_date": snapshotDateString(snapshot.snapshotDate),
            "parameters": parameterValueMap(output),
            "uncertainties": uncertaintyMap(output),
            "confidences": confidenceMap(output),
            "total_observations": output.totalObservations
        ]
    }

    private static func extractRecords(
        timeline: [GlucoseTimelineEvent],
        inferredMealEvents: [InferredMealEvent],
        readings: [GlucoseReading],
        therapyProfile: TherapyProfileContext?,
        cohort: String,
        clinicalSex: String,
        weightKg: Double?,
        availableUntil: Date
    ) -> [LearnerActionOutcomeRecord] {
        let startingPriors = clinicalStartingPriors(
            cohort: cohort,
            clinicalSex: clinicalSex,
            weightKg: weightKg
        )
        let referenceISF = therapyProfile?.insulinSensitivity ?? startingPriors.insulinSensitivityMgDlPerUnit
        let targetLow = therapyProfile?.targetLow ?? 90
        let targetHigh = therapyProfile?.targetHigh ?? 120
        let targetMidpoint = (targetLow + targetHigh) / 2.0
        let insulin = extractCorrectionISFRecords(
            timeline: timeline,
            readings: readings,
            availableUntil: availableUntil
        )
        let mealCR = extractCarbRatioRecords(
            timeline: timeline,
            readings: readings,
            availableUntil: availableUntil,
            referenceISF: referenceISF,
            targetGlucose: targetMidpoint
        )
        let mealAbsorption = extractCarbAbsorptionRecords(
            timeline: timeline,
            readings: readings,
            availableUntil: availableUntil
        )
        let exercise = extractExerciseRecords(
            timeline: timeline,
            readings: readings,
            availableUntil: availableUntil
        )
        let dawn = extractDawnRecords(
            timeline: timeline,
            readings: readings,
            availableUntil: availableUntil
        )

        return insulin + mealCR + mealAbsorption + exercise + dawn
    }

    private static func deriveObservations(
        from records: [LearnerActionOutcomeRecord]
    ) -> [LearnerObservation] {
        records.map { record in
            LearnerObservation(
                id: record.id,
                kind: record.kind,
                anchorEventID: record.anchorEventID,
                anchorTimestamp: record.anchorTimestamp,
                resolvedAt: record.resolvedAt,
                observedValue: record.derivedOutcomeValue,
                secondaryValue: record.secondaryInputValue,
                timeBucket: record.timeBucket,
                confidence: record.confidence,
                signals: record.signals
            )
        }
    }

    private static func extractCorrectionISFRecords(
        timeline: [GlucoseTimelineEvent],
        readings: [GlucoseReading],
        availableUntil: Date
    ) -> [LearnerActionOutcomeRecord] {
        let insulinEvents = timeline
            .filter { $0.kind == .insulin && ($0.value?.amount ?? 0) >= 0.5 }
            .sorted { $0.timestamp < $1.timestamp }

        return insulinEvents.compactMap { event in
            guard inferredInsulinType(for: event) == "bolus" else { return nil }

            let responseWindow: TimeInterval = 180 * 60
            let resolvedAt = event.timestamp.addingTimeInterval(responseWindow)
            guard resolvedAt <= availableUntil else { return nil }
            guard !containsConfoundingISFEvents(in: timeline, for: event, responseWindow: responseWindow) else { return nil }
            guard let units = event.value?.amount, units > 0 else { return nil }
            guard let before = reading(near: event.timestamp, toleranceMinutes: 15, readings: readings),
                  let after = reading(near: resolvedAt, toleranceMinutes: 20, readings: readings)
            else {
                return nil
            }
            guard !CGMArtifactDetector.containsLikelyArtifact(
                readings: readings,
                timeline: timeline,
                from: event.timestamp,
                to: resolvedAt
            ) else {
                return nil
            }

            let observedISF = (before.value - after.value) / units
            guard observedISF.isFinite, observedISF > 5, observedISF < 200 else { return nil }

            return LearnerActionOutcomeRecord(
                id: "learner:isf:\(event.dedupeKey)",
                kind: .correctionISF,
                anchorEventID: event.id,
                anchorTimestamp: event.timestamp,
                resolvedAt: resolvedAt,
                glucoseBefore: before.value,
                glucoseAfter: after.value,
                primaryInputValue: units,
                secondaryInputValue: nil,
                derivedOutcomeValue: observedISF,
                timeBucket: DateHelpers.timeBucket(for: event.timestamp).rawValue.lowercased(),
                confidence: .high,
                signals: signalContext(around: event.timestamp, timeline: timeline)
            )
        }
    }

    private static func extractCarbRatioRecords(
        timeline: [GlucoseTimelineEvent],
        readings: [GlucoseReading],
        availableUntil: Date,
        referenceISF: Double,
        targetGlucose: Double
    ) -> [LearnerActionOutcomeRecord] {
        let meals = timeline
            .filter {
                $0.kind == .meal
                    && qualifiesForMealLearning(
                        $0,
                        timeline: timeline,
                        readings: readings,
                        requireNearbyBolus: true
                    )
            }
            .sorted { $0.timestamp < $1.timestamp }

        return meals.compactMap { meal in
            let responseWindow: TimeInterval = 240 * 60
            let resolvedAt = meal.timestamp.addingTimeInterval(responseWindow)
            guard resolvedAt <= availableUntil else { return nil }
            guard !containsConfoundingMealResponseEvents(in: timeline, mealEvent: meal, analysisWindow: responseWindow, allowedMealBolusWindow: 30 * 60) else {
                return nil
            }
            guard let carbs = meal.value?.amount, carbs >= 10 else { return nil }

            let bolusUnits = timeline
                .filter {
                    $0.kind == .insulin
                        && abs($0.timestamp.timeIntervalSince(meal.timestamp)) <= 30 * 60
                        && inferredInsulinType(for: $0) == "bolus"
                }
                .compactMap(\.value?.amount)
                .reduce(0, +)
            guard bolusUnits >= 0.5 else { return nil }

            guard let before = reading(near: meal.timestamp, toleranceMinutes: 15, readings: readings),
                  let after = reading(near: resolvedAt, toleranceMinutes: 20, readings: readings)
            else {
                return nil
            }
            guard !CGMArtifactDetector.containsLikelyArtifact(
                readings: readings,
                timeline: timeline,
                from: meal.timestamp,
                to: resolvedAt
            ) else {
                return nil
            }

            let correctionComponent = before.value > targetGlucose + 20
                ? max(0, (before.value - targetGlucose) / max(referenceISF, 5))
                : 0
            var mealComponent = bolusUnits - correctionComponent
            guard mealComponent >= 0.3 else { return nil }

            let glucoseChange = after.value - before.value
            if abs(glucoseChange) > 10 {
                let insulinError = glucoseChange / max(referenceISF, 5)
                let corrected = mealComponent + insulinError
                if corrected >= 0.3 {
                    mealComponent = corrected
                }
            }

            let observedCR = carbs / mealComponent
            guard observedCR.isFinite, observedCR > 2, observedCR < 50 else { return nil }

            return LearnerActionOutcomeRecord(
                id: "learner:cr:\(meal.dedupeKey)",
                kind: .carbRatio,
                anchorEventID: meal.id,
                anchorTimestamp: meal.timestamp,
                resolvedAt: resolvedAt,
                glucoseBefore: before.value,
                glucoseAfter: after.value,
                primaryInputValue: bolusUnits,
                secondaryInputValue: carbs,
                derivedOutcomeValue: observedCR,
                timeBucket: DateHelpers.timeBucket(for: meal.timestamp).rawValue.lowercased(),
                confidence: learningConfidence(for: meal, fallback: .medium),
                signals: signalContext(around: meal.timestamp, timeline: timeline)
            )
        }
    }

    private static func extractCarbAbsorptionRecords(
        timeline: [GlucoseTimelineEvent],
        readings: [GlucoseReading],
        availableUntil: Date
    ) -> [LearnerActionOutcomeRecord] {
        let meals = timeline
            .filter {
                $0.kind == .meal
                    && qualifiesForMealLearning(
                        $0,
                        timeline: timeline,
                        readings: readings,
                        requireNearbyBolus: true
                    )
            }
            .sorted { $0.timestamp < $1.timestamp }

        return meals.compactMap { meal in
            let analysisWindow: TimeInterval = 5 * 3600
            let resolvedAt = meal.timestamp.addingTimeInterval(analysisWindow)
            guard resolvedAt <= availableUntil else { return nil }
            guard let observation = carbAbsorptionObservation(for: meal, readings: readings, timeline: timeline) else { return nil }
            guard !CGMArtifactDetector.containsLikelyArtifact(
                readings: readings,
                timeline: timeline,
                from: meal.timestamp,
                to: resolvedAt
            ) else {
                return nil
            }
            return LearnerActionOutcomeRecord(
                id: "learner:carb-absorption:\(meal.dedupeKey)",
                kind: .carbAbsorption,
                anchorEventID: meal.id,
                anchorTimestamp: meal.timestamp,
                resolvedAt: resolvedAt,
                glucoseBefore: observation.baseline,
                glucoseAfter: observation.peak,
                primaryInputValue: meal.value?.amount ?? 0,
                secondaryInputValue: meal.value?.amount,
                derivedOutcomeValue: observation.duration,
                timeBucket: DateHelpers.timeBucket(for: meal.timestamp).rawValue.lowercased(),
                confidence: learningConfidence(for: meal, fallback: .medium),
                signals: signalContext(around: meal.timestamp, timeline: timeline)
            )
        }
    }

    private static func extractExerciseRecords(
        timeline: [GlucoseTimelineEvent],
        readings: [GlucoseReading],
        availableUntil: Date
    ) -> [LearnerActionOutcomeRecord] {
        let exerciseEvents = timeline
            .filter { $0.kind == .exercise && qualifiesForExerciseLearning($0) }
            .sorted { $0.timestamp < $1.timestamp }

        return exerciseEvents.flatMap { event -> [LearnerActionOutcomeRecord] in
            let analysisWindow: TimeInterval = 12 * 3600
            let resolvedAt = event.timestamp.addingTimeInterval(analysisWindow)
            guard resolvedAt <= availableUntil else { return [] }
            guard !containsConfoundingExerciseResponseEvents(in: timeline, exerciseEvent: event, lookbackWindow: 90 * 60, analysisWindow: analysisWindow) else {
                return []
            }
            guard let baseline = readings.last(where: {
                $0.timestamp <= event.timestamp && $0.timestamp >= event.timestamp.addingTimeInterval(-45 * 60)
            }) else {
                return []
            }

            let postReadings = readings.filter {
                $0.timestamp >= event.timestamp && $0.timestamp <= resolvedAt
            }
            guard postReadings.count >= 3 else { return [] }
            guard !CGMArtifactDetector.containsLikelyArtifact(
                readings: readings,
                timeline: timeline,
                from: event.timestamp,
                to: resolvedAt
            ) else {
                return []
            }

            let changes = postReadings.map {
                (
                    timestamp: $0.timestamp,
                    elapsedHours: max($0.timestamp.timeIntervalSince(event.timestamp) / 3600.0, 0),
                    change: $0.value - baseline.value
                )
            }
            guard let nadir = changes.min(by: { $0.change < $1.change }),
                  nadir.change <= -10,
                  nadir.elapsedHours >= 1 else {
                return []
            }

            let reboundThreshold = max(abs(nadir.change) * 0.35, 8.0)
            guard let recovery = changes.first(where: {
                $0.timestamp > nadir.timestamp && $0.change >= -reboundThreshold
            }) else {
                return []
            }

            let workoutLoad = max(normalizedWorkoutLoad(from: event), 0.6)
            let decayHours = min(max(recovery.elapsedHours * min(max(workoutLoad, 0.85), 1.2), 4.0), 18.0)
            let boost = min(max(1.0 + (abs(nadir.change) / 120.0) * min(max(workoutLoad, 0.85), 1.15), 1.02), 1.35)
            let signals = signalContext(around: event.timestamp, timeline: timeline)

            return [
                LearnerActionOutcomeRecord(
                    id: "learner:exercise-boost:\(event.dedupeKey)",
                    kind: .exerciseBoost,
                    anchorEventID: event.id,
                    anchorTimestamp: event.timestamp,
                    resolvedAt: resolvedAt,
                    glucoseBefore: baseline.value,
                    glucoseAfter: baseline.value + nadir.change,
                    primaryInputValue: event.value?.amount ?? 0,
                    secondaryInputValue: abs(nadir.change),
                    derivedOutcomeValue: boost,
                    timeBucket: DateHelpers.timeBucket(for: event.timestamp).rawValue.lowercased(),
                    confidence: .medium,
                    signals: signals
                ),
                LearnerActionOutcomeRecord(
                    id: "learner:exercise-decay:\(event.dedupeKey)",
                    kind: .exerciseDecay,
                    anchorEventID: event.id,
                    anchorTimestamp: event.timestamp,
                    resolvedAt: resolvedAt,
                    glucoseBefore: baseline.value,
                    glucoseAfter: baseline.value + recovery.change,
                    primaryInputValue: event.value?.amount ?? 0,
                    secondaryInputValue: event.value?.amount,
                    derivedOutcomeValue: decayHours,
                    timeBucket: DateHelpers.timeBucket(for: event.timestamp).rawValue.lowercased(),
                    confidence: .medium,
                    signals: signals
                )
            ]
        }
    }

    private static func extractDawnRecords(
        timeline: [GlucoseTimelineEvent],
        readings: [GlucoseReading],
        availableUntil: Date
    ) -> [LearnerActionOutcomeRecord] {
        guard !readings.isEmpty else { return [] }
        let calendar = Calendar.current
        let groupedReadings = Dictionary(grouping: readings) { calendar.startOfDay(for: $0.timestamp) }

        return groupedReadings.keys.sorted().compactMap { dayStart in
            guard let dayReadings = groupedReadings[dayStart], !dayReadings.isEmpty else { return nil }
            let baseline = closestReading(
                in: dayReadings,
                from: dayStart,
                startHour: 3.0,
                endHour: 5.5,
                targetHour: 4.5
            )
            let dawn = closestReading(
                in: dayReadings,
                from: dayStart,
                startHour: 6.0,
                endHour: 8.5,
                targetHour: 7.5
            )
            guard let baseline, let dawn else { return nil }
            guard dawn.timestamp <= availableUntil else { return nil }
            guard !containsConfoundingDawnEvent(in: timeline, from: dayStart, to: dawn.timestamp) else { return nil }
            guard !CGMArtifactDetector.containsLikelyArtifact(
                readings: dayReadings,
                timeline: timeline,
                from: baseline.timestamp,
                to: dawn.timestamp
            ) else {
                return nil
            }

            let delta = dawn.value - baseline.value
            guard delta.isFinite, delta >= 6, delta <= 120 else { return nil }
            let id = "learner:dawn:\(snapshotDateString(dayStart))"
            return LearnerActionOutcomeRecord(
                id: id,
                kind: .dawnRise,
                anchorEventID: id,
                anchorTimestamp: baseline.timestamp,
                resolvedAt: dawn.timestamp,
                glucoseBefore: baseline.value,
                glucoseAfter: dawn.value,
                primaryInputValue: 0,
                secondaryInputValue: nil,
                derivedOutcomeValue: delta,
                timeBucket: "morning",
                confidence: .medium,
                signals: signalContext(around: dawn.timestamp, timeline: timeline)
            )
        }
    }

    private static func buildParameterOutput(
        observations: [LearnerObservation],
        therapyProfile: TherapyProfileContext?,
        feedQuality: InsightConfidence,
        cohort: String,
        clinicalSex: String,
        weightKg: Double?,
        generatedAt: Date,
        historicalSummary: HistoricalInsightWindowSummary?,
        statisticalAnalysis: StatisticalAnalysis?
    ) -> BayesianParameterOutput {
        let correctionObs = observations.filter { $0.kind == .correctionISF }
        let carbRatioObs = observations.filter { $0.kind == .carbRatio }
        let carbAbsorptionObs = observations.filter { $0.kind == .carbAbsorption }
        let exerciseBoostObs = observations.filter { $0.kind == .exerciseBoost }
        let exerciseDecayObs = observations.filter { $0.kind == .exerciseDecay }
        let dawnObs = observations.filter { $0.kind == .dawnRise }

        let referenceDate = generatedAt.addingTimeInterval(-7 * 24 * 3600)
        let observationsThisWeek = observations.filter { $0.resolvedAt >= referenceDate }.count
        let profileSignals = therapyProfileSignalCount(therapyProfile)
        let startingPriors = clinicalStartingPriors(
            cohort: cohort,
            clinicalSex: clinicalSex,
            weightKg: weightKg
        )
        let cohortISF = startingPriors.insulinSensitivityMgDlPerUnit

        let isfValues = correctionObs.map(\.observedValue)
        let isfBase = positivePosteriorEstimate(
            observations: isfValues,
            priorValue: therapyProfile?.insulinSensitivity ?? cohortISF,
            priorLogStd: 0.7,
            minValue: 10,
            maxValue: 200,
            feedQuality: feedQuality
        ) ?? therapyProfile?.insulinSensitivity.map {
            profileEstimate(
                value: $0,
                observationCount: max(correctionObs.count, 1),
                baseUncertainty: 10,
                minimumUncertainty: 3,
                feedQuality: feedQuality
            )
        }

        let timeModifiers = buildTimeBucketModifiers(
            observations: correctionObs,
            baseISF: isfBase?.value ?? therapyProfile?.insulinSensitivity ?? cohortISF,
            feedQuality: feedQuality
        )

        let carbRatioEstimate = positivePosteriorEstimate(
            observations: weightedMealObservationValues(carbRatioObs),
            priorValue: therapyProfile?.carbRatio ?? startingPriors.carbRatioGramsPerUnit,
            priorLogStd: 0.6,
            minValue: 2,
            maxValue: 50,
            feedQuality: feedQuality
        )

        let carbSensitivity: ParameterEstimate? = {
            guard let isf = isfBase?.value else { return nil }
            if let carbRatioEstimate {
                let value = isf / max(carbRatioEstimate.value, 0.01)
                let relativeVariance = pow((isfBase?.uncertainty ?? 0) / max(isf, 0.01), 2)
                    + pow((carbRatioEstimate.uncertainty ?? 0) / max(carbRatioEstimate.value, 0.01), 2)
                let sigma = value * sqrt(max(relativeVariance, 0))
                return posteriorEstimate(
                    value: value,
                    uncertainty: sigma,
                    rawConfidence: min(isfBase?.confidenceScore ?? 0.2, carbRatioEstimate.confidenceScore),
                    observationCount: carbRatioObs.count,
                    feedQuality: feedQuality
                )
            }
            return nil
        }()

        let carbAbsorption = boundedNormalPosteriorEstimate(
            observations: weightedMealObservationValues(carbAbsorptionObs),
            priorMean: 3.0,
            priorStd: 0.9,
            minimumValue: 1.5,
            maximumValue: 4.5,
            feedQuality: feedQuality
        )

        let weightedExerciseBoostObservations = weightedExerciseObservationValues(exerciseBoostObs)
        let weightedExerciseDecayObservations = weightedExerciseObservationValues(exerciseDecayObs)

        let exerciseDiscoveryEnabled = discoverySupportsModifier(
            statisticalAnalysis,
            variableKeys: [
                "exercise_minutes",
                "exercise_intensity_score",
                "exercise_whoop_strain",
                "exercise_avg_heart_rate_bpm",
                "exercise_max_heart_rate_bpm",
            ]
        )
        let mealTimingDiscoveryEffect = strongestDiscoveryEffect(
            statisticalAnalysis,
            variableKeys: [
                "pre_bolus_minutes",
                "meal_carbs_g",
            ]
        )
        let mealCompositionDiscoveryEffect = strongestDiscoveryEffect(
            statisticalAnalysis,
            variableKeys: [
                "fat_g",
                "protein_g",
            ]
        )
        let recoveryDiscoveryEffect = strongestDiscoveryEffect(
            statisticalAnalysis,
            variableKeys: [
                "sleep_hours_prev",
                "sleep_debt_hours",
                "sleep_quality_score",
                "sleep_deep_hours",
                "sleep_rem_hours",
                "sleep_awakenings",
                "resting_heart_rate_bpm",
                "hrv_ms",
            ]
        )
        let overnightDiscoveryEffect = strongestDiscoveryEffect(
            statisticalAnalysis,
            variableKeys: [
                "has_late_meal",
                "has_evening_insulin",
                "short_sleep_context",
            ]
        )
        let stressIllnessDiscoveryEffect = strongestDiscoveryEffect(
            statisticalAnalysis,
            variableKeys: [
                "stress_flags_24h",
                "stress_manual_score",
                "illness_flags_72h",
                "illness_flag",
                "alcohol_flag",
            ]
        )
        let weekendDiscoveryEffect = strongestDiscoveryEffect(
            statisticalAnalysis,
            variableKeys: ["weekend_flag"]
        )

        let exerciseBoost = exerciseDiscoveryEnabled ? positivePosteriorEstimate(
            observations: weightedExerciseBoostObservations,
            priorValue: 1.08,
            priorLogStd: 0.15,
            minValue: 1.0,
            maxValue: 1.4,
            feedQuality: feedQuality
        ) : nil

        let exerciseDecay = exerciseDiscoveryEnabled ? boundedNormalPosteriorEstimate(
            observations: weightedExerciseDecayObservations,
            priorMean: 8.0,
            priorStd: 2.5,
            minimumValue: 4.0,
            maximumValue: 18.0,
            feedQuality: feedQuality
        ) : nil

        let dawn = boundedNormalPosteriorEstimate(
            observations: dawnObs.map(\.observedValue),
            priorMean: 18.0,
            priorStd: 15.0,
            minimumValue: 6.0,
            maximumValue: 120.0,
            feedQuality: feedQuality
        )

        let evidence = ParameterObservationEvidence(
            insulinResponseWindows: correctionObs.count,
            mealResponseWindows: max(carbRatioObs.count, carbAbsorptionObs.count),
            exerciseResponseWindows: max(exerciseBoostObs.count, exerciseDecayObs.count),
            overnightTrendWindows: dawnObs.count,
            therapyProfileSignals: profileSignals,
            feedDataQuality: feedQuality
        )
        let gatedFamilies = buildGatedParameterFamilies(
            statisticalAnalysis: statisticalAnalysis,
            carbAbsorption: carbAbsorption,
            exerciseBoost: exerciseBoost,
            exerciseDecay: exerciseDecay,
            exerciseObservationCount: max(exerciseBoostObs.count, exerciseDecayObs.count),
            mealTimingDiscoveryEffect: mealTimingDiscoveryEffect,
            mealCompositionDiscoveryEffect: mealCompositionDiscoveryEffect,
            recoveryDiscoveryEffect: recoveryDiscoveryEffect,
            overnightDiscoveryEffect: overnightDiscoveryEffect,
            stressIllnessDiscoveryEffect: stressIllnessDiscoveryEffect,
            weekendDiscoveryEffect: weekendDiscoveryEffect
        )
        let activeModifiers = buildActiveModifiers(
            gatedFamilies: gatedFamilies,
            carbAbsorption: carbAbsorption,
            exerciseBoost: exerciseBoost,
            exerciseDecay: exerciseDecay
        )

        return BayesianParameterOutput(
            schemaVersion: 2,
            generatedAt: generatedAt,
            totalObservations: observations.count + profileSignals,
            observationsThisWeek: observationsThisWeek,
            lastUpdated: generatedAt,
            evidence: evidence,
            isfBaseline: isfBase,
            isfTimeModifiers: timeModifiers,
            carbSensitivity: carbSensitivity,
            carbAbsorptionHours: carbAbsorption,
            exerciseSensitivityBoost: exerciseBoost,
            exerciseEffectDecayHours: exerciseDecay,
            dawnEffectMagnitude: dawn,
            gatedFamilies: gatedFamilies,
            activeModifiers: activeModifiers
        )
    }

    private static func discoverySupportsModifier(
        _ statisticalAnalysis: StatisticalAnalysis?,
        variableKeys: [String]
    ) -> Bool {
        strongestDiscoveryEffect(statisticalAnalysis, variableKeys: variableKeys) != nil
    }

    private static func strongestDiscoveryEffect(
        _ statisticalAnalysis: StatisticalAnalysis?,
        variableKeys: [String]
    ) -> ValidatedEffect? {
        guard let statisticalAnalysis else { return nil }
        let keySet = Set(variableKeys)
        return statisticalAnalysis.validatedEffects
            .filter {
                keySet.contains($0.variableKey)
                    && confidenceRank($0.confidence) >= 2
                    && ($0.stabilityScore ?? 0) >= 0.5
            }
            .sorted { lhs, rhs in
                let lhsRank = confidenceRank(lhs.confidence)
                let rhsRank = confidenceRank(rhs.confidence)
                if lhsRank != rhsRank {
                    return lhsRank > rhsRank
                }
                let lhsStability = lhs.stabilityScore ?? 0
                let rhsStability = rhs.stabilityScore ?? 0
                if lhsStability != rhsStability {
                    return lhsStability > rhsStability
                }
                if lhs.sampleSize != rhs.sampleSize {
                    return lhs.sampleSize > rhs.sampleSize
                }
                return abs(lhs.effectSize ?? 0) > abs(rhs.effectSize ?? 0)
            }
            .first
    }

    private static func strongestInteractionEffect(
        _ statisticalAnalysis: StatisticalAnalysis?,
        variableKeys: [String],
        windowKinds: [AnalysisWindowKind] = []
    ) -> InteractionEffect? {
        guard let statisticalAnalysis else { return nil }
        let keySet = Set(variableKeys)
        let allowedWindowKinds = Set(windowKinds)
        return statisticalAnalysis.interactionEffects
            .filter { effect in
                !Set(effect.variableKeys).isDisjoint(with: keySet)
                    && (allowedWindowKinds.isEmpty || allowedWindowKinds.contains(effect.windowKind))
                    && confidenceRank(effect.confidence) >= 2
                    && (effect.stabilityScore ?? 0) >= 0.5
            }
            .sorted { lhs, rhs in
                let lhsRank = confidenceRank(lhs.confidence)
                let rhsRank = confidenceRank(rhs.confidence)
                if lhsRank != rhsRank {
                    return lhsRank > rhsRank
                }
                let lhsStability = lhs.stabilityScore ?? 0
                let rhsStability = rhs.stabilityScore ?? 0
                if lhsStability != rhsStability {
                    return lhsStability > rhsStability
                }
                if lhs.sampleSize != rhs.sampleSize {
                    return lhs.sampleSize > rhs.sampleSize
                }
                return (lhs.rSquared ?? 0) > (rhs.rSquared ?? 0)
            }
            .first
    }

    private static func buildGatedParameterFamilies(
        statisticalAnalysis: StatisticalAnalysis?,
        carbAbsorption: ParameterEstimate?,
        exerciseBoost: ParameterEstimate?,
        exerciseDecay: ParameterEstimate?,
        exerciseObservationCount: Int,
        mealTimingDiscoveryEffect: ValidatedEffect?,
        mealCompositionDiscoveryEffect: ValidatedEffect?,
        recoveryDiscoveryEffect: ValidatedEffect?,
        overnightDiscoveryEffect: ValidatedEffect?,
        stressIllnessDiscoveryEffect: ValidatedEffect?,
        weekendDiscoveryEffect: ValidatedEffect?
    ) -> [GatedParameterFamilyEstimate] {
        var families: [GatedParameterFamilyEstimate] = []

        let mealTimingInteraction = strongestInteractionEffect(
            statisticalAnalysis,
            variableKeys: ["meal_carbs_g", "pre_bolus_minutes"],
            windowKinds: [.mealResponse]
        )
        if let mealTimingDiscoveryEffect {
            let confidenceScore = max(
                confidenceToScore(mealTimingDiscoveryEffect.confidence),
                interactionConfidenceScore(mealTimingInteraction)
            )
            let summaryParts = [
                mealTimingDiscoveryEffect.summary,
                mealTimingInteraction?.summary,
            ].compactMap { $0 }
            families.append(
                GatedParameterFamilyEstimate(
                    id: "gated_family:meal_timing_effect",
                    key: "meal_timing_effect",
                    label: "Meal timing effect",
                    driverVariableKeys: ["meal_carbs_g", "pre_bolus_minutes"],
                    windowKinds: [AnalysisWindowKind.mealResponse.rawValue],
                    effectSize: mealTimingInteraction?.rSquared ?? mealTimingDiscoveryEffect.effectSize,
                    ciLower: mealTimingDiscoveryEffect.ciLower,
                    ciUpper: mealTimingDiscoveryEffect.ciUpper,
                    confidence: insightConfidence(fromScore: confidenceScore),
                    confidenceScore: confidenceScore,
                    sampleSize: max(mealTimingDiscoveryEffect.sampleSize, mealTimingInteraction?.sampleSize ?? 0),
                    summary: summaryParts.joined(separator: " "),
                    source: .historicalTrend
                )
            )
        }

        let mealCompositionInteraction = strongestInteractionEffect(
            statisticalAnalysis,
            variableKeys: ["fat_g", "protein_g", "meal_carbs_g"],
            windowKinds: [.mealResponse]
        )
        if mealCompositionDiscoveryEffect != nil || mealCompositionInteraction != nil || carbAbsorption != nil {
            let hasCompositionSignal = mealCompositionDiscoveryEffect != nil || mealCompositionInteraction != nil
            let confidenceScore = max(
                confidenceToScore(mealCompositionDiscoveryEffect?.confidence ?? .low),
                interactionConfidenceScore(mealCompositionInteraction),
                carbAbsorption?.confidenceScore ?? 0
            )
            var summaryParts: [String] = []
            if let carbAbsorption {
                summaryParts.append("Meal responses are currently unfolding over about \(String(format: "%.1f", carbAbsorption.value)) hours.")
            }
            if let mealCompositionDiscoveryEffect {
                summaryParts.append(mealCompositionDiscoveryEffect.summary)
            }
            if let mealCompositionInteraction {
                summaryParts.append(mealCompositionInteraction.summary)
            }
            families.append(
                GatedParameterFamilyEstimate(
                    id: "gated_family:meal_composition_effect",
                    key: "meal_composition_effect",
                    label: hasCompositionSignal ? "Meal composition effect" : "Meal response window",
                    driverVariableKeys: ["fat_g", "protein_g", "meal_carbs_g"],
                    windowKinds: [AnalysisWindowKind.mealResponse.rawValue],
                    effectSize: mealCompositionInteraction?.rSquared ?? mealCompositionDiscoveryEffect?.effectSize ?? carbAbsorption?.value,
                    ciLower: mealCompositionDiscoveryEffect?.ciLower,
                    ciUpper: mealCompositionDiscoveryEffect?.ciUpper,
                    confidence: insightConfidence(fromScore: confidenceScore),
                    confidenceScore: confidenceScore,
                    sampleSize: max(
                        mealCompositionDiscoveryEffect?.sampleSize ?? 0,
                        mealCompositionInteraction?.sampleSize ?? 0,
                        carbAbsorption?.observationCount ?? 0
                    ),
                    summary: summaryParts.joined(separator: " "),
                    source: carbAbsorption != nil ? .bayesianPosterior : .historicalTrend
                )
            )
        }

        let exerciseInteraction = strongestInteractionEffect(
            statisticalAnalysis,
            variableKeys: [
                "exercise_minutes",
                "exercise_intensity_score",
                "exercise_whoop_strain",
                "exercise_avg_heart_rate_bpm",
                "exercise_max_heart_rate_bpm",
            ],
            windowKinds: [.exerciseAftereffect]
        )
        if let exerciseBoost, let exerciseDecay {
            let confidenceScore = max(
                min(exerciseBoost.confidenceScore, exerciseDecay.confidenceScore),
                interactionConfidenceScore(exerciseInteraction)
            )
            let summaryParts = [
                "Exercise tends to increase sensitivity to about \(String(format: "%.2f", exerciseBoost.value))x baseline for roughly \(String(format: "%.1f", exerciseDecay.value)) hours.",
                exerciseInteraction?.summary,
            ].compactMap { $0 }
            families.append(
                GatedParameterFamilyEstimate(
                    id: "gated_family:exercise_effect",
                    key: "exercise_effect",
                    label: "Exercise recovery effect",
                    driverVariableKeys: [
                        "exercise_minutes",
                        "exercise_intensity_score",
                        "exercise_whoop_strain",
                        "exercise_avg_heart_rate_bpm",
                        "exercise_max_heart_rate_bpm",
                    ],
                    windowKinds: [AnalysisWindowKind.exerciseAftereffect.rawValue],
                    effectSize: exerciseInteraction?.rSquared ?? exerciseBoost.value,
                    ciLower: nil,
                    ciUpper: nil,
                    confidence: insightConfidence(fromScore: confidenceScore),
                    confidenceScore: confidenceScore,
                    sampleSize: max(exerciseObservationCount, exerciseInteraction?.sampleSize ?? 0),
                    summary: summaryParts.joined(separator: " "),
                    source: .bayesianPosterior
                )
            )
        }

        let recoveryInteraction = strongestInteractionEffect(
            statisticalAnalysis,
            variableKeys: [
                "sleep_hours_prev",
                "sleep_debt_hours",
                "sleep_quality_score",
                "sleep_deep_hours",
                "sleep_rem_hours",
                "sleep_awakenings",
                "resting_heart_rate_bpm",
                "hrv_ms",
            ],
            windowKinds: [.wholeDaySummary, .overnight]
        )
        if let recoveryDiscoveryEffect {
            let confidenceScore = max(
                confidenceToScore(recoveryDiscoveryEffect.confidence),
                interactionConfidenceScore(recoveryInteraction)
            )
            let summaryParts = [
                recoveryDiscoveryEffect.summary,
                recoveryInteraction?.summary,
            ].compactMap { $0 }
            families.append(
                GatedParameterFamilyEstimate(
                    id: "gated_family:recovery_effect",
                    key: "recovery_effect",
                    label: modifierLabel(for: recoveryDiscoveryEffect.variableKey, fallback: "Recovery-linked modifier"),
                    driverVariableKeys: [
                        "sleep_hours_prev",
                        "sleep_debt_hours",
                        "sleep_quality_score",
                        "sleep_deep_hours",
                        "sleep_rem_hours",
                        "sleep_awakenings",
                        "resting_heart_rate_bpm",
                        "hrv_ms",
                    ],
                    windowKinds: [AnalysisWindowKind.wholeDaySummary.rawValue, AnalysisWindowKind.overnight.rawValue],
                    effectSize: recoveryInteraction?.rSquared ?? recoveryDiscoveryEffect.effectSize,
                    ciLower: recoveryDiscoveryEffect.ciLower,
                    ciUpper: recoveryDiscoveryEffect.ciUpper,
                    confidence: insightConfidence(fromScore: confidenceScore),
                    confidenceScore: confidenceScore,
                    sampleSize: max(recoveryDiscoveryEffect.sampleSize, recoveryInteraction?.sampleSize ?? 0),
                    summary: summaryParts.joined(separator: " "),
                    source: .historicalTrend
                )
            )
        }

        let overnightInteraction = strongestInteractionEffect(
            statisticalAnalysis,
            variableKeys: ["has_late_meal", "has_evening_insulin", "short_sleep_context"],
            windowKinds: [.overnight]
        )
        if let overnightDiscoveryEffect {
            let confidenceScore = max(
                confidenceToScore(overnightDiscoveryEffect.confidence),
                interactionConfidenceScore(overnightInteraction)
            )
            let summaryParts = [
                overnightDiscoveryEffect.summary,
                overnightInteraction?.summary,
            ].compactMap { $0 }
            families.append(
                GatedParameterFamilyEstimate(
                    id: "gated_family:overnight_carryover",
                    key: "overnight_carryover",
                    label: "Overnight carryover pattern",
                    driverVariableKeys: ["has_late_meal", "has_evening_insulin", "short_sleep_context"],
                    windowKinds: [AnalysisWindowKind.overnight.rawValue],
                    effectSize: overnightInteraction?.rSquared ?? overnightDiscoveryEffect.effectSize,
                    ciLower: overnightDiscoveryEffect.ciLower,
                    ciUpper: overnightDiscoveryEffect.ciUpper,
                    confidence: insightConfidence(fromScore: confidenceScore),
                    confidenceScore: confidenceScore,
                    sampleSize: max(overnightDiscoveryEffect.sampleSize, overnightInteraction?.sampleSize ?? 0),
                    summary: summaryParts.joined(separator: " "),
                    source: .historicalTrend
                )
            )
        }

        if let stressIllnessDiscoveryEffect {
            families.append(
                GatedParameterFamilyEstimate(
                    id: "gated_family:stress_illness_effect",
                    key: "stress_illness_effect",
                    label: modifierLabel(for: stressIllnessDiscoveryEffect.variableKey, fallback: "Stress or illness context"),
                    driverVariableKeys: ["stress_flags_24h", "stress_manual_score", "illness_flags_72h", "illness_flag", "alcohol_flag"],
                    windowKinds: [AnalysisWindowKind.wholeDaySummary.rawValue, AnalysisWindowKind.overnight.rawValue],
                    effectSize: stressIllnessDiscoveryEffect.effectSize,
                    ciLower: stressIllnessDiscoveryEffect.ciLower,
                    ciUpper: stressIllnessDiscoveryEffect.ciUpper,
                    confidence: stressIllnessDiscoveryEffect.confidence,
                    confidenceScore: confidenceToScore(stressIllnessDiscoveryEffect.confidence),
                    sampleSize: stressIllnessDiscoveryEffect.sampleSize,
                    summary: stressIllnessDiscoveryEffect.summary,
                    source: .historicalTrend
                )
            )
        }

        if let weekendDiscoveryEffect {
            families.append(
                GatedParameterFamilyEstimate(
                    id: "gated_family:weekend_effect",
                    key: "weekend_effect",
                    label: "Weekend modifier",
                    driverVariableKeys: ["weekend_flag"],
                    windowKinds: [AnalysisWindowKind.wholeDaySummary.rawValue],
                    effectSize: weekendDiscoveryEffect.effectSize,
                    ciLower: weekendDiscoveryEffect.ciLower,
                    ciUpper: weekendDiscoveryEffect.ciUpper,
                    confidence: weekendDiscoveryEffect.confidence,
                    confidenceScore: confidenceToScore(weekendDiscoveryEffect.confidence),
                    sampleSize: weekendDiscoveryEffect.sampleSize,
                    summary: weekendDiscoveryEffect.summary,
                    source: .historicalTrend
                )
            )
        }

        return families.sorted { lhs, rhs in
            if lhs.confidenceScore != rhs.confidenceScore {
                return lhs.confidenceScore > rhs.confidenceScore
            }
            return lhs.sampleSize > rhs.sampleSize
        }
    }

    private static func buildActiveModifiers(
        gatedFamilies: [GatedParameterFamilyEstimate],
        carbAbsorption: ParameterEstimate?,
        exerciseBoost: ParameterEstimate?,
        exerciseDecay: ParameterEstimate?
    ) -> [ActiveModifierEstimate] {
        let modifiers = gatedFamilies.map { family in
            let effectDescription: String
            switch family.key {
            case "meal_timing_effect":
                effectDescription = family.summary
            case "meal_composition_effect":
                effectDescription = family.summary
            case "exercise_effect":
                if let exerciseBoost, let exerciseDecay {
                    effectDescription = "Exercise tends to increase sensitivity to about \(String(format: "%.2f", exerciseBoost.value))x baseline for roughly \(String(format: "%.1f", exerciseDecay.value)) hours. \(family.summary)"
                } else {
                    effectDescription = family.summary
                }
            default:
                effectDescription = family.summary
            }

            return ActiveModifierEstimate(
                id: "active_modifier:\(family.key)",
                key: family.key,
                label: family.label,
                effectDescription: effectDescription,
                confidence: family.confidence,
                confidenceScore: family.confidenceScore,
                observationCount: family.sampleSize,
                source: family.source
            )
        }

        return modifiers.sorted { lhs, rhs in
            if lhs.confidenceScore != rhs.confidenceScore {
                return lhs.confidenceScore > rhs.confidenceScore
            }
            return lhs.observationCount > rhs.observationCount
        }
    }

    private static func interactionConfidenceScore(_ interaction: InteractionEffect?) -> Double {
        guard let interaction else { return 0 }
        let confidenceComponent = confidenceToScore(interaction.confidence)
        let rSquaredComponent = min(max(interaction.rSquared ?? 0, 0), 1)
        let stabilityComponent = min(max(interaction.stabilityScore ?? 0, 0), 1)
        return (confidenceComponent * 0.45) + (rSquaredComponent * 0.35) + (stabilityComponent * 0.20)
    }

    private static func modifierLabel(for variableKey: String, fallback: String) -> String {
        switch variableKey {
        case "sleep_hours_prev", "sleep_debt_hours", "sleep_quality_score", "sleep_deep_hours", "sleep_rem_hours", "sleep_awakenings":
            return "Sleep-linked modifier"
        case "resting_heart_rate_bpm", "hrv_ms":
            return "Recovery-linked modifier"
        case "stress_flags_24h", "stress_manual_score":
            return "Stress-linked modifier"
        case "illness_flags_72h", "illness_flag":
            return "Illness-linked modifier"
        case "alcohol_flag":
            return "Alcohol-linked modifier"
        default:
            return fallback
        }
    }

    private static func buildTimeBucketModifiers(
        observations: [LearnerObservation],
        baseISF: Double,
        feedQuality: InsightConfidence
    ) -> [TimeBucketParameterEstimate] {
        let grouped = Dictionary(grouping: observations, by: { $0.timeBucket ?? "night" })
        return TimeBucket.allCases.map { bucket in
            let key = bucket.rawValue.lowercased()
            let observedMultipliers = (grouped[key] ?? [])
                .map { $0.observedValue / max(baseISF, 0.01) }
                .filter { $0.isFinite && $0 > 0 }

            guard !observedMultipliers.isEmpty else {
                return TimeBucketParameterEstimate(
                    id: key,
                    bucket: key,
                    multiplier: 1.0,
                    confidence: .low,
                    confidenceScore: 0.12,
                    observationCount: 0,
                    source: .therapyProfile
                )
            }

            let logs = observedMultipliers.map(Foundation.log)
            let priorStd = 0.2
            let posterior = normalPosterior(
                observations: logs,
                priorMean: 0,
                priorStd: priorStd,
                observedFloorStd: 0.18
            )
            let multiplier = min(max(Foundation.exp(posterior.mean), 0.7), 1.4)
            let rawConfidence = max(0, min(1, 1 - (posterior.std / priorStd)))
            let confidenceScore = min(max(rawConfidence * feedFactor(feedQuality), 0.12), 0.95)

            return TimeBucketParameterEstimate(
                id: key,
                bucket: key,
                multiplier: (multiplier * 100).rounded() / 100,
                confidence: confidence(score: confidenceScore),
                confidenceScore: confidenceScore,
                observationCount: observedMultipliers.count,
                source: .bayesianPosterior
            )
        }
    }

    private static func recordSnapshot(
        parameterOutput: BayesianParameterOutput?,
        existing: [LocalParameterSnapshot],
        referenceTime: Date
    ) -> [LocalParameterSnapshot] {
        guard let parameterOutput else {
            return existing.sorted { $0.snapshotDate > $1.snapshotDate }
        }

        let snapshotDate = Calendar.current.startOfDay(for: referenceTime)
        let snapshotID = snapshotDateString(snapshotDate)
        let incoming = LocalParameterSnapshot(
            id: snapshotID,
            snapshotDate: snapshotDate,
            recordedAt: referenceTime,
            parameterOutput: parameterOutput,
            syncedAt: existing.first(where: { $0.id == snapshotID })?.syncedAt
        )

        var merged = Dictionary(uniqueKeysWithValues: existing.map { ($0.id, $0) })
        merged[snapshotID] = incoming
        return merged.values
            .sorted { $0.snapshotDate > $1.snapshotDate }
            .prefix(maxSnapshots)
            .map { $0 }
    }

    private static func positivePosteriorEstimate(
        observations: [Double],
        priorValue: Double,
        priorLogStd: Double,
        minValue: Double,
        maxValue: Double,
        feedQuality: InsightConfidence
    ) -> ParameterEstimate? {
        let clean = observations.filter { $0.isFinite && $0 >= minValue && $0 <= maxValue }
        guard !clean.isEmpty else { return nil }

        let logs = clean.map(Foundation.log)
        let posterior = normalPosterior(
            observations: logs,
            priorMean: Foundation.log(max(priorValue, minValue)),
            priorStd: priorLogStd,
            observedFloorStd: 0.18
        )
        let value = min(max(Foundation.exp(posterior.mean), minValue), maxValue)
        let uncertainty = value * posterior.std
        let rawConfidence = max(0, min(1, 1 - (posterior.std / priorLogStd)))
        return posteriorEstimate(
            value: value,
            uncertainty: uncertainty,
            rawConfidence: rawConfidence,
            observationCount: clean.count,
            feedQuality: feedQuality
        )
    }

    private static func boundedNormalPosteriorEstimate(
        observations: [Double],
        priorMean: Double,
        priorStd: Double,
        minimumValue: Double,
        maximumValue: Double,
        feedQuality: InsightConfidence
    ) -> ParameterEstimate? {
        let clean = observations.filter { $0.isFinite && $0 >= minimumValue && $0 <= maximumValue }
        guard !clean.isEmpty else { return nil }

        let posterior = normalPosterior(
            observations: clean,
            priorMean: priorMean,
            priorStd: priorStd,
            observedFloorStd: max(priorStd * 0.35, 0.3)
        )
        let value = min(max(posterior.mean, minimumValue), maximumValue)
        let rawConfidence = max(0, min(1, 1 - (posterior.std / priorStd)))
        return posteriorEstimate(
            value: value,
            uncertainty: posterior.std,
            rawConfidence: rawConfidence,
            observationCount: clean.count,
            feedQuality: feedQuality
        )
    }

    private static func profileEstimate(
        value: Double,
        observationCount: Int,
        baseUncertainty: Double,
        minimumUncertainty: Double,
        feedQuality: InsightConfidence
    ) -> ParameterEstimate {
        let uncertainty = max(baseUncertainty / sqrt(Double(max(observationCount, 1))), minimumUncertainty)
        let uncertaintyRatio = max(0, min(1, 1 - (uncertainty / max(baseUncertainty, 0.001))))
        let observationScore = min(Double(observationCount) / 12.0, 1.0)
        let rawScore = (0.5 * observationScore) + (0.25 * uncertaintyRatio) + (0.25 * feedFactor(feedQuality))
        let confidenceScore = min(max(rawScore, 0.18), 0.85)

        return ParameterEstimate(
            value: value,
            uncertainty: uncertainty,
            confidence: confidence(score: confidenceScore),
            confidenceScore: confidenceScore,
            observationCount: observationCount,
            source: .therapyProfile
        )
    }

    private static func posteriorEstimate(
        value: Double,
        uncertainty: Double?,
        rawConfidence: Double,
        observationCount: Int,
        feedQuality: InsightConfidence
    ) -> ParameterEstimate {
        let confidenceScore = min(max(rawConfidence * feedFactor(feedQuality), 0.12), 0.98)
        return ParameterEstimate(
            value: value,
            uncertainty: uncertainty,
            confidence: confidence(score: confidenceScore),
            confidenceScore: confidenceScore,
            observationCount: observationCount,
            source: .bayesianPosterior
        )
    }

    private static func normalPosterior(
        observations: [Double],
        priorMean: Double,
        priorStd: Double,
        observedFloorStd: Double
    ) -> (mean: Double, std: Double) {
        let priorVariance = priorStd * priorStd
        let observedStd = max(sampleStandardDeviation(observations), observedFloorStd)
        let observedVariance = observedStd * observedStd
        let precision = (1.0 / priorVariance) + (Double(observations.count) / observedVariance)
        let posteriorVariance = 1.0 / max(precision, 1e-9)
        let posteriorMean = posteriorVariance * (
            (priorMean / priorVariance) + (observations.reduce(0, +) / observedVariance)
        )
        return (posteriorMean, sqrt(max(posteriorVariance, 0)))
    }

    private static func signalContext(
        around timestamp: Date,
        timeline: [GlucoseTimelineEvent]
    ) -> LearnerObservationSignals {
        let recentExercise = timeline.filter {
            $0.kind == .exercise
                && $0.timestamp <= timestamp
                && timestamp.timeIntervalSince($0.timestamp) <= 6 * 3600
        }
        let recentSteps = timeline.filter {
            $0.kind == .steps
                && $0.timestamp <= timestamp
                && timestamp.timeIntervalSince($0.timestamp) <= 6 * 3600
        }
        let recentSleep = timeline
            .filter {
                $0.kind == .sleep
                    && $0.timestamp <= timestamp
                    && timestamp.timeIntervalSince($0.timestamp) <= 36 * 3600
            }
            .sorted { $0.timestamp > $1.timestamp }
        let recentNotes = timeline.filter {
            $0.kind == .note
                && $0.timestamp <= timestamp
                && timestamp.timeIntervalSince($0.timestamp) <= 72 * 3600
        }

        let exerciseSummaries = recentExercise.compactMap(workoutSummary(from:))
        let strongestWorkout = exerciseSummaries.max { lhs, rhs in
            normalizedWorkoutLoad(from: lhs) < normalizedWorkoutLoad(from: rhs)
        }
        let workoutLoad = recentExercise.reduce(0.0) { partialResult, event in
            partialResult + normalizedWorkoutLoad(from: event)
        }
        let steps = recentSteps.compactMap(\.value?.amount).reduce(0, +)
        let sleepHours = recentSleep.first?.value?.amount ?? 0
        let stressFlags = recentNotes.filter { noteHasStressFlag($0) }.count
        let illnessFlags = recentNotes.filter { noteHasIllnessFlag($0) }.count
        let workoutIntensityScore = strongestWorkout.map { normalizedIntensityScore(for: $0) } ?? 0
        let incidentalStepLoad = max(steps - 2500, 0) / 6000.0
        let activityLevel = min(max(workoutLoad + incidentalStepLoad, 0), 2.0)
        let sleepQuality = sleepHours > 0 ? min(max(sleepHours / 8.0, 0.2), 1.2) : 0
        let stressLevel = min(Double(stressFlags) * 0.5, 1.5)
        let illnessFlag = illnessFlags > 0 ? 1.0 : 0.0

        return LearnerObservationSignals(
            activityLevel: activityLevel,
            sleepQuality: sleepQuality,
            stressLevel: stressLevel,
            illnessFlag: illnessFlag,
            stepsLast6h: steps,
            workoutIntensityScore: workoutIntensityScore,
            workoutMinutesLast6h: strongestWorkout?.durationMinutes,
            workoutWhoopStrain: strongestWorkout?.whoopStrain,
            workoutType: strongestWorkout?.activityType
        )
    }

    private static func reading(
        near target: Date,
        toleranceMinutes: Double,
        readings: [GlucoseReading]
    ) -> GlucoseReading? {
        let tolerance = toleranceMinutes * 60
        return readings
            .filter { abs($0.timestamp.timeIntervalSince(target)) <= tolerance }
            .min(by: { abs($0.timestamp.timeIntervalSince(target)) < abs($1.timestamp.timeIntervalSince(target)) })
    }

    private static func qualifiesForMealLearning(
        _ meal: GlucoseTimelineEvent,
        timeline: [GlucoseTimelineEvent],
        readings: [GlucoseReading],
        requireNearbyBolus: Bool
    ) -> Bool {
        guard let carbs = meal.value?.amount,
              carbs >= 10,
              carbs <= 160
        else {
            return false
        }

        guard !textSuggestsRescueCarbs(meal),
              !hasNearDuplicateMeal(meal, timeline: timeline)
        else {
            return false
        }

        guard let baseline = reading(near: meal.timestamp, toleranceMinutes: 15, readings: readings),
              baseline.value >= 70,
              baseline.value <= 250
        else {
            return false
        }

        if requireNearbyBolus {
            return nearbyMealBolusUnits(for: meal, timeline: timeline) >= 0.5
        }

        return true
    }

    private static func nearbyMealBolusUnits(
        for meal: GlucoseTimelineEvent,
        timeline: [GlucoseTimelineEvent]
    ) -> Double {
        timeline
            .filter {
                $0.kind == .insulin
                    && abs($0.timestamp.timeIntervalSince(meal.timestamp)) <= 30 * 60
                    && inferredInsulinType(for: $0) == "bolus"
            }
            .compactMap(\.value?.amount)
            .reduce(0, +)
    }

    private static func hasNearDuplicateMeal(
        _ meal: GlucoseTimelineEvent,
        timeline: [GlucoseTimelineEvent]
    ) -> Bool {
        timeline.contains {
            $0.kind == .meal
                && $0.dedupeKey != meal.dedupeKey
                && abs($0.timestamp.timeIntervalSince(meal.timestamp)) <= 15 * 60
                && abs(($0.value?.amount ?? 0) - (meal.value?.amount ?? 0)) <= 2
        }
    }

    private static func textSuggestsRescueCarbs(_ event: GlucoseTimelineEvent) -> Bool {
        let text = "\(event.title) \(event.detail ?? "") \(event.metadata?.values.joined(separator: " ") ?? "")"
            .lowercased()
        return [
            "hypo",
            "rescue",
            "treatment",
            "treat low",
            "low bg",
            "low glucose",
            "juice",
            "glucose tab",
            "glucotab"
        ].contains { text.contains($0) }
    }

    private static func containsConfoundingISFEvents(
        in timeline: [GlucoseTimelineEvent],
        for insulinEvent: GlucoseTimelineEvent,
        responseWindow: TimeInterval
    ) -> Bool {
        let eventTime = insulinEvent.timestamp
        let responseEnd = eventTime.addingTimeInterval(responseWindow)
        let mealExerciseLookback: TimeInterval = 60 * 60
        let insulinLookback: TimeInterval = 180 * 60

        return timeline.contains { event in
            guard event.id != insulinEvent.id else {
                return false
            }

            switch event.kind {
            case .meal, .exercise:
                return event.timestamp >= eventTime.addingTimeInterval(-mealExerciseLookback)
                    && event.timestamp <= responseEnd
            case .insulin:
                return inferredInsulinType(for: event) == "bolus"
                    && event.timestamp >= eventTime.addingTimeInterval(-insulinLookback)
                    && event.timestamp <= responseEnd
            default:
                return false
            }
        }
    }

    private static func containsConfoundingMealResponseEvents(
        in timeline: [GlucoseTimelineEvent],
        mealEvent: GlucoseTimelineEvent,
        analysisWindow: TimeInterval,
        allowedMealBolusWindow: TimeInterval
    ) -> Bool {
        timeline.contains { event in
            guard event.id != mealEvent.id,
                  event.timestamp > mealEvent.timestamp,
                  event.timestamp <= mealEvent.timestamp.addingTimeInterval(analysisWindow)
            else {
                return false
            }

            switch event.kind {
            case .meal, .exercise:
                return true
            case .insulin:
                return event.timestamp.timeIntervalSince(mealEvent.timestamp) > allowedMealBolusWindow
            default:
                return false
            }
        }
    }

    private static func containsConfoundingExerciseResponseEvents(
        in timeline: [GlucoseTimelineEvent],
        exerciseEvent: GlucoseTimelineEvent,
        lookbackWindow: TimeInterval,
        analysisWindow: TimeInterval
    ) -> Bool {
        timeline.contains { event in
            guard event.id != exerciseEvent.id else { return false }

            let inLookback = event.timestamp < exerciseEvent.timestamp
                && event.timestamp >= exerciseEvent.timestamp.addingTimeInterval(-lookbackWindow)
            let inWindow = event.timestamp > exerciseEvent.timestamp
                && event.timestamp <= exerciseEvent.timestamp.addingTimeInterval(analysisWindow)
            guard inLookback || inWindow else { return false }

            switch event.kind {
            case .meal, .insulin, .exercise:
                return true
            default:
                return false
            }
        }
    }

    private static func containsConfoundingDawnEvent(
        in timeline: [GlucoseTimelineEvent],
        from start: Date,
        to end: Date
    ) -> Bool {
        timeline.contains { event in
            guard event.timestamp >= start, event.timestamp <= end else { return false }
            switch event.kind {
            case .meal, .insulin, .exercise:
                return true
            default:
                return false
            }
        }
    }

    private static func carbAbsorptionObservation(
        for mealEvent: GlucoseTimelineEvent,
        readings: [GlucoseReading],
        timeline: [GlucoseTimelineEvent]
    ) -> (duration: Double, baseline: Double, peak: Double)? {
        let analysisWindow: TimeInterval = 5 * 3600
        let baselineWindow: TimeInterval = 45 * 60
        let allowedMealBolusWindow: TimeInterval = 30 * 60

        guard !containsConfoundingMealResponseEvents(
            in: timeline,
            mealEvent: mealEvent,
            analysisWindow: analysisWindow,
            allowedMealBolusWindow: allowedMealBolusWindow
        ) else {
            return nil
        }

        guard let baseline = readings.last(where: {
            $0.timestamp <= mealEvent.timestamp && $0.timestamp >= mealEvent.timestamp.addingTimeInterval(-baselineWindow)
        }) else {
            return nil
        }

        let postReadings = readings.filter {
            $0.timestamp >= mealEvent.timestamp && $0.timestamp <= mealEvent.timestamp.addingTimeInterval(analysisWindow)
        }
        guard postReadings.count >= 3 else { return nil }

        let rises = postReadings.map {
            (
                elapsedHours: max($0.timestamp.timeIntervalSince(mealEvent.timestamp) / 3600.0, 0),
                rise: $0.value - baseline.value
            )
        }

        guard let peak = rises.max(by: { $0.rise < $1.rise }),
              peak.rise.isFinite,
              peak.rise >= 12.0,
              peak.elapsedHours >= 0.75 else {
            return nil
        }

        let settleThreshold = max(peak.rise * 0.35, 8.0)
        let settledHour = rises.first(where: {
            $0.elapsedHours >= peak.elapsedHours && $0.rise <= settleThreshold
        })?.elapsedHours
        let duration = settledHour ?? peak.elapsedHours
        guard duration.isFinite, duration >= 1.0 else { return nil }
        return (
            duration: min(max(duration, 1.5), 4.5),
            baseline: baseline.value,
            peak: baseline.value + peak.rise
        )
    }

    private static func closestReading(
        in readings: [GlucoseReading],
        from dayStart: Date,
        startHour: Double,
        endHour: Double,
        targetHour: Double
    ) -> GlucoseReading? {
        let rangeStart = dayStart.addingTimeInterval(startHour * 3600.0)
        let rangeEnd = dayStart.addingTimeInterval(endHour * 3600.0)
        let target = dayStart.addingTimeInterval(targetHour * 3600.0)

        return readings
            .filter { $0.timestamp >= rangeStart && $0.timestamp <= rangeEnd }
            .min(by: {
                abs($0.timestamp.timeIntervalSince(target)) < abs($1.timestamp.timeIntervalSince(target))
            })
    }

    private static func inferredInsulinType(for event: GlucoseTimelineEvent) -> String {
        let detail = "\(event.title) \(event.detail ?? "")".lowercased()
        if detail.contains("basal")
            || detail.contains("long")
            || detail.contains("lantus")
            || detail.contains("tresiba")
            || detail.contains("levemir") {
            return "basal"
        }
        return "bolus"
    }

    private static func parseWorkoutIntensity(from detail: String?, metadata: [String: String]? = nil) -> String? {
        if let metadataIntensity = metadata?["intensity"]?.lowercased(),
           ["vigorous", "moderate", "light"].contains(metadataIntensity) {
            return metadataIntensity
        }
        guard let detail = detail?.lowercased() else { return nil }
        if detail.contains("vigorous") { return "vigorous" }
        if detail.contains("moderate") { return "moderate" }
        if detail.contains("light") { return "light" }
        return nil
    }

    private static func parseWorkoutIntensityScore(from detail: String?, metadata: [String: String]? = nil) -> Double? {
        switch parseWorkoutIntensity(from: detail, metadata: metadata) {
        case "vigorous":
            return 1.0
        case "moderate":
            return 0.65
        case "light":
            return 0.35
        default:
            return nil
        }
    }

    private static func normalizedIntensityScore(for workout: WorkoutSignalSummary) -> Double {
        guard let score = parseWorkoutIntensityScore(from: workout.intensity, metadata: ["intensity": workout.intensity ?? ""]) else {
            return 0.45
        }
        return score
    }

    private static func normalizedWorkoutLoad(from event: GlucoseTimelineEvent) -> Double {
        guard let summary = workoutSummary(from: event) else { return 0 }
        return normalizedWorkoutLoad(from: summary)
    }

    private static func normalizedWorkoutLoad(from workout: WorkoutSignalSummary) -> Double {
        let durationFactor = max(workout.durationMinutes / 30.0, 0.25)
        let intensityFactor = max(normalizedIntensityScore(for: workout), 0.35)
        let strainFactor: Double = {
            guard let strain = workout.whoopStrain else { return 1.0 }
            return min(max(strain / 12.0, 0.7), 1.45)
        }()
        let typeFactor = workoutTypeFactor(workout.activityType)
        return durationFactor * intensityFactor * strainFactor * typeFactor
    }

    private static func workoutTypeFactor(_ activityType: String?) -> Double {
        guard let activityType = activityType?.lowercased() else { return 1.0 }
        if ["running", "cycling", "swimming", "soccer", "football", "basketball", "rowing", "hiking"].contains(where: activityType.contains) {
            return 1.1
        }
        if ["walking", "yoga", "pilates", "stretching"].contains(where: activityType.contains) {
            return 0.75
        }
        if ["strength", "weights", "resistance", "crossfit"].contains(where: activityType.contains) {
            return 0.95
        }
        return 1.0
    }

    private static func qualifiesForExerciseLearning(_ event: GlucoseTimelineEvent) -> Bool {
        guard let summary = workoutSummary(from: event) else { return false }
        let intensityScore = normalizedIntensityScore(for: summary)
        let loadScore = normalizedWorkoutLoad(from: summary)
        if summary.durationMinutes >= 12 && intensityScore >= 0.5 {
            return true
        }
        if let strain = summary.whoopStrain, strain >= 5 {
            return true
        }
        return loadScore >= 0.7
    }

    private static func weightedExerciseObservationValues(_ observations: [LearnerObservation]) -> [Double] {
        observations.flatMap { observation in
            let weightScore =
                (observation.signals.activityLevel * 0.45)
                + (observation.signals.workoutIntensityScore * 0.35)
                + (((observation.signals.workoutWhoopStrain ?? 0) / 12.0) * 0.2)
            let repeats = min(max(Int((weightScore * 2.5).rounded(.awayFromZero)), 1), 3)
            return Array(repeating: observation.observedValue, count: repeats)
        }
    }

    private static func weightedMealObservationValues(_ observations: [LearnerObservation]) -> [Double] {
        observations.flatMap { observation in
            let repeats: Int
            switch observation.confidence {
            case .high:
                repeats = 3
            case .medium:
                repeats = 2
            case .low:
                repeats = 1
            }
            return Array(repeating: observation.observedValue, count: repeats)
        }
    }

    private static func learningConfidence(
        for mealEvent: GlucoseTimelineEvent,
        fallback: InsightConfidence
    ) -> InsightConfidence {
        fallback == .low ? .medium : fallback
    }

    private static func noteHasStressFlag(_ event: GlucoseTimelineEvent) -> Bool {
        if event.metadata?["tag_stress"] == "true" {
            return true
        }
        let haystack = "\(event.title) \(event.detail ?? "")".lowercased()
        let keywords = ["stress", "stressed", "anxious", "anxiety", "exam", "deadline", "panic"]
        return keywords.contains(where: { haystack.contains($0) })
    }

    private static func noteHasIllnessFlag(_ event: GlucoseTimelineEvent) -> Bool {
        if event.metadata?["tag_illness"] == "true" {
            return true
        }
        let haystack = "\(event.title) \(event.detail ?? "")".lowercased()
        let keywords = ["sick", "ill", "illness", "fever", "flu", "cold", "infection", "covid"]
        return keywords.contains(where: { haystack.contains($0) })
    }

    private static func sampleStandardDeviation(_ values: [Double]) -> Double {
        guard values.count > 1 else { return 0 }
        let mean = values.reduce(0, +) / Double(values.count)
        let variance = values.reduce(0) { partial, value in
            partial + pow(value - mean, 2)
        } / Double(values.count - 1)
        return sqrt(max(variance, 0))
    }

    private static func average(_ values: [Double]) -> Double? {
        guard !values.isEmpty else { return nil }
        return values.reduce(0, +) / Double(values.count)
    }

    private static func sleepSessionSummary(from event: GlucoseTimelineEvent) -> SleepSessionSignal? {
        let end = metadataDate(from: event, key: "sessionEnd") ?? event.timestamp
        let totalHours = metadataDouble(from: event, key: "sessionHours") ?? event.value?.amount
        guard let totalHours, totalHours > 0 else { return nil }
        let start = metadataDate(from: event, key: "sessionStart") ?? end.addingTimeInterval(-totalHours * 3600.0)
        return SleepSessionSignal(
            start: start,
            end: end,
            totalHours: totalHours,
            deepHours: metadataDouble(from: event, key: "deepSleepHours"),
            remHours: metadataDouble(from: event, key: "remSleepHours"),
            awakeningCount: metadataString(from: event, key: "awakeningCount").flatMap(Int.init),
            qualityScore: metadataDouble(from: event, key: "qualityScore"),
            source: metadataString(from: event, key: "sourceName") ?? event.source?.displayName
        )
    }

    private static func workoutSummary(from event: GlucoseTimelineEvent) -> WorkoutSignalSummary? {
        guard event.kind == .exercise else { return nil }
        let durationMinutes = metadataDouble(from: event, key: "durationMinutes") ?? event.value?.amount
        guard let durationMinutes, durationMinutes > 0 else { return nil }
        let start = metadataDate(from: event, key: "startDate") ?? event.timestamp
        return WorkoutSignalSummary(
            activityType: workoutType(from: event) ?? event.title,
            intensity: parseWorkoutIntensity(from: event.detail, metadata: event.metadata),
            durationMinutes: durationMinutes,
            start: start,
            end: metadataDate(from: event, key: "endDate"),
            source: metadataString(from: event, key: "sourceName") ?? event.source?.displayName,
            energyKilocalories: metadataDouble(from: event, key: "energyKilocalories"),
            distanceMeters: metadataDouble(from: event, key: "distanceMeters"),
            isIndoor: metadataBool(from: event, key: "isIndoor"),
            brandName: metadataString(from: event, key: "brandName"),
            whoopStrain: metadataDouble(from: event, key: "whoopStrain"),
            avgHeartRateBpm: metadataDouble(from: event, key: "avgHeartRateBpm"),
            maxHeartRateBpm: metadataDouble(from: event, key: "maxHeartRateBpm")
        )
    }

    private static func workoutType(from event: GlucoseTimelineEvent) -> String? {
        if let metadataType = metadataString(from: event, key: "activityType"), !metadataType.isEmpty {
            return metadataType
        }
        guard let detail = event.detail else { return nil }
        return detail.components(separatedBy: "•")
            .first?
            .trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private static func contextMetricValue(
        from events: [GlucoseTimelineEvent],
        metricKey: String
    ) -> Double? {
        events.first(where: { metadataString(from: $0, key: "metric") == metricKey })?.value?.amount
    }

    private static func metadataString(from event: GlucoseTimelineEvent, key: String) -> String? {
        guard let raw = event.metadata?[key]?.trimmingCharacters(in: .whitespacesAndNewlines),
              !raw.isEmpty else {
            return nil
        }
        return raw
    }

    private static func metadataDouble(from event: GlucoseTimelineEvent, key: String) -> Double? {
        guard let raw = metadataString(from: event, key: key) else { return nil }
        return Double(raw)
    }

    private static func metadataDate(from event: GlucoseTimelineEvent, key: String) -> Date? {
        guard let raw = metadataString(from: event, key: key) else { return nil }
        return ISO8601DateFormatter.cachedGlucaFormatter.date(from: raw)
    }

    private static func metadataBool(from event: GlucoseTimelineEvent, key: String) -> Bool? {
        guard let raw = metadataString(from: event, key: key)?.lowercased() else { return nil }
        switch raw {
        case "true", "1", "yes":
            return true
        case "false", "0", "no":
            return false
        default:
            return nil
        }
    }

    private static func therapyProfileSignalCount(_ profile: TherapyProfileContext?) -> Int {
        guard let profile else { return 0 }
        return [
            profile.diaHours,
            profile.carbRatio,
            profile.insulinSensitivity,
            profile.targetLow,
            profile.targetHigh,
            profile.basalRateUnitsPerHour
        ]
        .compactMap { $0 }
        .count
    }

    private static func clinicalStartingPriors(
        cohort: String,
        clinicalSex: String,
        weightKg: Double?
    ) -> ClinicalStartingPriors {
        let resolvedCohort = ClinicalCohort(rawValue: cohort.lowercased()) ?? .adult
        let resolvedSex = ClinicalSex(rawValue: clinicalSex.lowercased()) ?? .unspecified
        return ClinicalPriorCalculator.startingPriors(
            cohort: resolvedCohort,
            sex: resolvedSex,
            weightKg: weightKg
        )
    }

    private static func confidence(score: Double) -> InsightConfidence {
        switch score {
        case 0.7...:
            return .high
        case 0.45...:
            return .medium
        default:
            return .low
        }
    }

    private static func confidenceToScore(_ confidence: InsightConfidence) -> Double {
        switch confidence {
        case .high:
            return 0.82
        case .medium:
            return 0.62
        case .low:
            return 0.32
        }
    }

    private static func insightConfidence(fromScore score: Double) -> InsightConfidence {
        switch score {
        case 0.7...:
            return .high
        case 0.45...:
            return .medium
        default:
            return .low
        }
    }

    private static func confidenceRank(_ confidence: InsightConfidence) -> Int {
        switch confidence {
        case .high:
            return 3
        case .medium:
            return 2
        case .low:
            return 1
        }
    }

    private static func feedFactor(_ confidence: InsightConfidence) -> Double {
        switch confidence {
        case .high:
            return 0.95
        case .medium:
            return 0.78
        case .low:
            return 0.55
        }
    }

    private static func snapshotDateString(_ date: Date) -> String {
        let formatter = DateFormatter()
        formatter.calendar = Calendar(identifier: .iso8601)
        formatter.locale = Locale(identifier: "en_US_POSIX")
        formatter.timeZone = TimeZone(secondsFromGMT: 0)
        formatter.dateFormat = "yyyy-MM-dd"
        return formatter.string(from: date)
    }

    private static func parameterValueMap(_ output: BayesianParameterOutput) -> [String: Any] {
        [
            "isf_baseline": output.isfBaseline?.value as Any,
            "isf_baseline_source": output.isfBaseline?.source.rawValue as Any,
            "isf_time_modifiers": Dictionary(uniqueKeysWithValues: output.isfTimeModifiers.map {
                ($0.bucket, ["multiplier": $0.multiplier, "source": $0.source.rawValue] as [String: Any])
            }),
            "carb_sensitivity": output.carbSensitivity?.value as Any,
            "carb_sensitivity_source": output.carbSensitivity?.source.rawValue as Any,
            "carb_absorption_hours": output.carbAbsorptionHours?.value as Any,
            "carb_absorption_hours_source": output.carbAbsorptionHours?.source.rawValue as Any,
            "exercise_sensitivity_boost": output.exerciseSensitivityBoost?.value as Any,
            "exercise_sensitivity_boost_source": output.exerciseSensitivityBoost?.source.rawValue as Any,
            "exercise_effect_decay_hours": output.exerciseEffectDecayHours?.value as Any,
            "exercise_effect_decay_hours_source": output.exerciseEffectDecayHours?.source.rawValue as Any,
            "dawn_effect_magnitude": output.dawnEffectMagnitude?.value as Any,
            "dawn_effect_magnitude_source": output.dawnEffectMagnitude?.source.rawValue as Any
        ]
    }

    private static func uncertaintyMap(_ output: BayesianParameterOutput) -> [String: Any] {
        [
            "isf_baseline": output.isfBaseline?.uncertainty as Any,
            "carb_sensitivity": output.carbSensitivity?.uncertainty as Any,
            "carb_absorption_hours": output.carbAbsorptionHours?.uncertainty as Any,
            "exercise_sensitivity_boost": output.exerciseSensitivityBoost?.uncertainty as Any,
            "exercise_effect_decay_hours": output.exerciseEffectDecayHours?.uncertainty as Any,
            "dawn_effect_magnitude": output.dawnEffectMagnitude?.uncertainty as Any
        ]
    }

    private static func confidenceMap(_ output: BayesianParameterOutput) -> [String: Any] {
        [
            "isf_baseline": ["label": output.isfBaseline?.confidence.rawValue as Any, "score": output.isfBaseline?.confidenceScore as Any],
            "carb_sensitivity": ["label": output.carbSensitivity?.confidence.rawValue as Any, "score": output.carbSensitivity?.confidenceScore as Any],
            "carb_absorption_hours": ["label": output.carbAbsorptionHours?.confidence.rawValue as Any, "score": output.carbAbsorptionHours?.confidenceScore as Any],
            "exercise_sensitivity_boost": ["label": output.exerciseSensitivityBoost?.confidence.rawValue as Any, "score": output.exerciseSensitivityBoost?.confidenceScore as Any],
            "exercise_effect_decay_hours": ["label": output.exerciseEffectDecayHours?.confidence.rawValue as Any, "score": output.exerciseEffectDecayHours?.confidenceScore as Any],
            "dawn_effect_magnitude": ["label": output.dawnEffectMagnitude?.confidence.rawValue as Any, "score": output.dawnEffectMagnitude?.confidenceScore as Any]
        ]
    }
}

private extension ISO8601DateFormatter {
    static let cachedGlucaFormatter: ISO8601DateFormatter = {
        let formatter = ISO8601DateFormatter()
        formatter.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        return formatter
    }()
}
