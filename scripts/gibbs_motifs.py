import copy
import random

random.seed(9875)
def generateAllKmers(dnas, k):
    motifs = {}
    for index, dna in enumerate(dnas):
        all_kmers = []
        for i in range(len(dna) - k + 1):
            all_kmers.append(dna[i:i+k])
        motifs[index] = all_kmers
    return motifs

def selectRandomMotif(dnas, k):
    all_possible_kmers = generateAllKmers(dnas, k)
    motifs = []
    for index, dna in enumerate(dnas):
        motifs.append(random.choice(all_possible_kmers[index]))
    return motifs

def initializeProfile(k, nucleotides):
    profile = {}
    for nucleotide in nucleotides:
        profile[nucleotide] = [0]*k
    return profile

def calculateCountMatrix(profile_kmers, k, nucleotides):
    profile = initializeProfile(k, nucleotides)
    for i in range(k):
        for profile_kmer in profile_kmers:
            profile[profile_kmer[i]][i] += 1
    return profile

def addPseudocounts(profile):
    for key, value in profile.items():
        profile[key] = [x+1 for x in value]
    return profile

def calculateProfile(profile_kmers, k, nucleotides):
    n = len(profile_kmers)
    profile = calculateCountMatrix(profile_kmers, k, nucleotides)
    profile = addPseudocounts(profile)
    for nucleotide in nucleotides:
        profile[nucleotide] = [x / (len(nucleotides)+n) for x in profile[nucleotide]]
    return profile

def calculateBestPorbableKmer(dna, profile, k):
    if len(profile['A']) != k:
        return "profile probability cannot be calculated because K does not match the profile length"
    max_prob = -999
    kmers = []
    weights = []
    for i in range(len(dna) - k + 1):
        kmer = dna[i:i+k]
        probability = 0
        for index, value in enumerate(kmer):
            if index == 0:
                probability = profile[value][index]
            else:
                probability *= profile[value][index]
        kmers.append(kmer)
        weights.append(probability)
    total = sum(weights)
    weights = [x / total for x in weights]
    return random.choices(kmers, weights, k=1)[0]

def findIdealMotifFromCountMatrix(countMatrix, k, nucleotides):
    idealMotif = ''
    for i in range(k):
        maxValue = -999
        idealNucleotide = ''
        for nucleotide in nucleotides:
            if maxValue < countMatrix[nucleotide][i]:
                maxValue = countMatrix[nucleotide][i]
                idealNucleotide = nucleotide
        idealMotif += idealNucleotide
    return idealMotif

def hamming(p1, p2):
    d = 0
    for i in range(len(p1)):
        if p1[i] != p2[i]:
            d += 1
    return d

def scoreMotifs(profile_kmers, k, nucleotides):
    countMatrix = calculateCountMatrix(profile_kmers, k, nucleotides)
    idealMotif = findIdealMotifFromCountMatrix(countMatrix, k, nucleotides)
    score = 0
    for kmer in profile_kmers:
        score += hamming(idealMotif, kmer)
    return score

def getNucleotides(t):
    all_nucleotides = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
    return ''.join(all_nucleotides[0:t])

def gibbsMotif(dnas, k, t, N, nucleotides):
    score = float('inf')
    best_motifs = selectRandomMotif(dnas, k)
    gibbs_motifs = copy.deepcopy(best_motifs)
    weights = [1]*t
    for i in range(N):
        poped_index = random.choices(range(t), weights, k=1)[0]
        poped_value = gibbs_motifs.pop(poped_index)
        profile = calculateProfile(gibbs_motifs, k, nucleotides)
        new_kmer = calculateBestPorbableKmer(dnas[poped_index], profile, k)
        gibbs_motifs.insert(poped_index, new_kmer)
        calculate_motifs_score = scoreMotifs(gibbs_motifs, k, nucleotides)
        best_motifs_score = scoreMotifs(best_motifs, k, nucleotides)
        if calculate_motifs_score < best_motifs_score:
            best_motifs = copy.deepcopy(gibbs_motifs)
            score = calculate_motifs_score
    return best_motifs, score

def getBestMotif(motifs, k, nucleotides):
    countMatrix = calculateCountMatrix(motifs, k, nucleotides)
    return findIdealMotifFromCountMatrix(countMatrix, k, nucleotides)

def isValidBestMotif(best_motif):
    for nucleotide in best_motif:
        if best_motif.count(nucleotide) > 1:
            return False
    return True

def getBestMotifsOfVaryingLenght(dnas):
    t = len(dnas)
    N = 100
    return_results = []
    nucleotides = getNucleotides(t)
    for k in range(2, t):
        score = float('inf')
        motifs = []
        for ittrate in range(1000):
            results = gibbsMotif(dnas, k, t, N, nucleotides)
            if score > results[1]:
                score = results[1]
                motifs = results[0]
        best_motif = getBestMotif(motifs, k, nucleotides)
        if isValidBestMotif(best_motif):
            return_results.append(best_motif)
    return return_results

def getPCAGroups(columns, lengths):
    return_result = []
    for l in lengths:
        dna_results, matrix_results = generateDNAString(columns, l)
        for index, i in enumerate(dna_results):
            return_result.extend(convertDNAToPCAGroup(dna_results[index][0], getBestMotifsOfVaryingLenght(i[1])))
    return return_result