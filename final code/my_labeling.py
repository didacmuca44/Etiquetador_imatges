__authors__ = [str(1745518), str(1748251), str(1746018)]
__group__ = 'Team_64'

from utils_data import read_dataset, read_extended_dataset, crop_images, visualize_retrieval

import numpy as np
import time
import matplotlib.pyplot as plt

from Kmeans import KMeans, get_colors
from KNN import KNN


def get_color_labels(images, K=3, options=None, remove_bg=False, bg_threshold=40, min_percentage=None):
    """
    Etiqueta colors amb KMeans.

    Millores:
    - remove_bg=True: elimina aproximadament el fons mirant les cantonades.
    - min_percentage: elimina colors poc representatius.
    """
    color_labels = []
    color_percentages = []

    for img in images:

        data = img

        if remove_bg:
            img_array = np.array(img, dtype=float)

            h = img_array.shape[0]
            w = img_array.shape[1]

            corners = np.array([
                img_array[0, 0],
                img_array[0, w - 1],
                img_array[h - 1, 0],
                img_array[h - 1, w - 1]
            ])

            background_color = np.mean(corners, axis=0)

            pixels = img_array.reshape(-1, img_array.shape[-1])
            dist = np.sqrt(np.sum((pixels - background_color) ** 2, axis=1))

            object_pixels = pixels[dist > bg_threshold]

            if len(object_pixels) > 0:
                data = object_pixels
            else:
                data = pixels

        km = KMeans(data, K, options)
        km.fit()

        centroid_colors = get_colors(km.centroids)

        labels = km.labels
        total = len(labels)

        percentages = {}

        for i in range(len(centroid_colors)):
            color = centroid_colors[i]
            count = np.sum(labels == i)

            if color not in percentages:
                percentages[color] = 0

            percentages[color] += count / total

        ordered = sorted(percentages.items(), key=lambda x: x[1], reverse=True)

        img_colors = []

        for color, perc in ordered:
            if min_percentage is None or perc >= min_percentage:
                img_colors.append(color)

        if len(img_colors) == 0 and len(ordered) > 0:
            img_colors.append(ordered[0][0])

        color_labels.append(img_colors)
        color_percentages.append(percentages)

    return color_labels, color_percentages


def get_shape_labels(train_imgs, train_labels, test_imgs, k=3, options=None):
    """
    Etiqueta formes amb KNN.
    """
    knn = KNN(train_imgs, train_labels, options)
    predictions = knn.predict(test_imgs, k)

    return predictions


def get_shape_labels_with_confidence(train_imgs, train_labels, test_imgs, k=3, options=None):
    """
    Etiqueta formes amb KNN i retorna també la confiança.
    """
    knn = KNN(train_imgs, train_labels, options)
    predictions, confidences = knn.predict_with_confidence(test_imgs, k)

    return predictions, confidences


def Retrieval_by_color(images, color_labels, question, color_percentages=None):
    """
    Retorna imatges que contenen el color o colors demanats.
    Compara sense distingir majúscules/minúscules.
    """
    if type(question) == str:
        question_list = [question.lower()]
    else:
        question_list = []
        for q in question:
            question_list.append(str(q).lower())

    result = []
    scores = []

    for i in range(len(images)):

        colors_img = color_labels[i]

        if type(colors_img) == str:
            colors_img_list = [colors_img.lower()]
        else:
            colors_img_list = []
            for c in colors_img:
                colors_img_list.append(str(c).lower())

        ok = True

        for color in question_list:
            if color not in colors_img_list:
                ok = False
                break

        if ok:
            result.append(images[i])

            score = 0

            if color_percentages is not None:
                percentages_norm = {}

                for c, p in color_percentages[i].items():
                    percentages_norm[str(c).lower()] = p

                for color in question_list:
                    if color in percentages_norm:
                        score += percentages_norm[color]

            scores.append(score)

    if color_percentages is not None:
        order = np.argsort(scores)[::-1]

        ordered_result = []
        for idx in order:
            ordered_result.append(result[idx])

        return ordered_result

    return result


def Retrieval_by_shape(images, shape_labels, question):
    """
    Retorna imatges que tenen la forma demanada.
    """
    result = []
    question = str(question).lower()

    for i in range(len(images)):
        if str(shape_labels[i]).lower() == question:
            result.append(images[i])

    return result


def Retrieval_combined(images, shape_labels, color_labels, shape_question, color_question):
    """
    Retorna imatges que coincideixen amb forma i color.
    """
    shape_question = str(shape_question).lower()

    if type(color_question) == str:
        color_question_list = [color_question.lower()]
    else:
        color_question_list = []
        for c in color_question:
            color_question_list.append(str(c).lower())

    result = []

    for i in range(len(images)):

        shape_ok = str(shape_labels[i]).lower() == shape_question

        colors_img = color_labels[i]

        if type(colors_img) == str:
            colors_img_list = [colors_img.lower()]
        else:
            colors_img_list = []
            for c in colors_img:
                colors_img_list.append(str(c).lower())

        color_ok = True

        for color in color_question_list:
            if color not in colors_img_list:
                color_ok = False
                break

        if shape_ok and color_ok:
            result.append(images[i])

    return result


def Retrieval_combined_ranked(images, shape_labels, color_labels, shape_question, color_question,
                              color_percentages=None, shape_confidences=None):
    """
    Millora pròpia:
    Retrieval combinat ordenat per score.

    score = 0.6 * confiança_forma + 0.4 * percentatge_color
    """
    shape_question = str(shape_question).lower()

    if type(color_question) == str:
        color_question_list = [color_question.lower()]
    else:
        color_question_list = []
        for c in color_question:
            color_question_list.append(str(c).lower())

    result = []
    scores = []

    for i in range(len(images)):

        if str(shape_labels[i]).lower() != shape_question:
            continue

        colors_img = color_labels[i]

        if type(colors_img) == str:
            colors_img_list = [colors_img.lower()]
        else:
            colors_img_list = []
            for c in colors_img:
                colors_img_list.append(str(c).lower())

        color_ok = True

        for color in color_question_list:
            if color not in colors_img_list:
                color_ok = False
                break

        if not color_ok:
            continue

        color_score = 0

        if color_percentages is not None:
            percentages_norm = {}

            for c, p in color_percentages[i].items():
                percentages_norm[str(c).lower()] = p

            for color in color_question_list:
                if color in percentages_norm:
                    color_score += percentages_norm[color]
        else:
            color_score = 1

        if shape_confidences is not None:
            shape_score = shape_confidences[i]
        else:
            shape_score = 1

        score = 0.6 * shape_score + 0.4 * color_score

        result.append(images[i])
        scores.append(score)

    order = np.argsort(scores)[::-1]

    ordered_result = []
    ordered_scores = []

    for idx in order:
        ordered_result.append(result[idx])
        ordered_scores.append(scores[idx])

    return ordered_result, ordered_scores


def Get_shape_accuracy(predicted_labels, gt_labels):
    """
    Accuracy clàssic per forma.
    """
    if len(predicted_labels) == 0:
        return 0

    correctes = 0

    for i in range(len(predicted_labels)):
        if predicted_labels[i] == gt_labels[i]:
            correctes += 1

    return correctes / len(predicted_labels)


def Get_color_accuracy(predicted_labels, gt_labels):
    """
    Accuracy de color amb Jaccard.
    """
    if len(predicted_labels) == 0:
        return 0

    total = 0

    for i in range(len(predicted_labels)):

        pred = predicted_labels[i]
        gt = gt_labels[i]

        if type(pred) == str:
            pred_list = [pred.lower()]
        else:
            pred_list = []
            for p in pred:
                pred_list.append(str(p).lower())

        if type(gt) == str:
            gt_list = [gt.lower()]
        else:
            gt_list = []
            for g in gt:
                gt_list.append(str(g).lower())

        pred_set = set(pred_list)
        gt_set = set(gt_list)

        if len(pred_set) == 0 and len(gt_set) == 0:
            score = 1
        else:
            intersection = pred_set.intersection(gt_set)
            union = pred_set.union(gt_set)

            if len(union) == 0:
                score = 0
            else:
                score = len(intersection) / len(union)

        total += score

    return total / len(predicted_labels)


def Kmean_statistics(image, Kmax, options=None):
    """
    Executa KMeans per K=2 fins Kmax i calcula:
    - WCD
    - iteracions
    - temps
    """
    Ks = []
    WCDs = []
    iterations = []
    times = []

    for k in range(2, Kmax + 1):
        km = KMeans(image, k, options)

        start = time.time()
        km.fit()
        end = time.time()

        wcd = km.withinClassDistance()

        Ks.append(k)
        WCDs.append(wcd)
        iterations.append(km.num_iter)
        times.append(end - start)

    results = {
        'K': Ks,
        'WCD': WCDs,
        'iterations': iterations,
        'time': times
    }

    plt.figure()
    plt.plot(Ks, WCDs, marker='o')
    plt.xlabel('K')
    plt.ylabel('WCD')
    plt.title('KMeans: WCD segons K')
    plt.show()

    plt.figure()
    plt.plot(Ks, iterations, marker='o')
    plt.xlabel('K')
    plt.ylabel('Iteracions')
    plt.title('KMeans: iteracions segons K')
    plt.show()

    plt.figure()
    plt.plot(Ks, times, marker='o')
    plt.xlabel('K')
    plt.ylabel('Temps')
    plt.title('KMeans: temps segons K')
    plt.show()

    return results


def compare_knn_options(train_imgs, train_class_labels, test_imgs, test_class_labels):
    """
    Compara configuracions de KNN.
    """
    experiments = []

    configs = [
        {'feature_type': 'pixels', 'distance': 'euclidean', 'voting': 'normal'},
        {'feature_type': 'pixels', 'distance': 'manhattan', 'voting': 'normal'},
        {'feature_type': 'simple', 'distance': 'euclidean', 'voting': 'normal'},
        {'feature_type': 'simple', 'distance': 'manhattan', 'voting': 'normal'},
        {'feature_type': 'grid', 'distance': 'manhattan', 'voting': 'weighted'}
    ]

    for config in configs:
        predictions = get_shape_labels(
            train_imgs,
            train_class_labels,
            test_imgs,
            k=3,
            options=config
        )

        acc = Get_shape_accuracy(predictions, test_class_labels)

        experiments.append({
            'config': config,
            'accuracy': acc
        })

    return experiments


def compare_kmeans_options(images, gt_color_labels, K=3):
    """
    Compara configuracions de KMeans.
    """
    experiments = []

    configs = [
        {'km_init': 'first', 'tolerance': 1, 'max_iter': 50, 'fitting': 'WCD'},
        {'km_init': 'random', 'tolerance': 1, 'max_iter': 50, 'fitting': 'WCD'},
        {'km_init': 'kmeans++', 'tolerance': 1, 'max_iter': 50, 'fitting': 'WCD'}
    ]

    for config in configs:
        pred_colors, percentages = get_color_labels(
            images,
            K=K,
            options=config,
            remove_bg=False
        )

        acc = Get_color_accuracy(pred_colors, gt_color_labels)

        experiments.append({
            'config': config,
            'accuracy': acc
        })

    return experiments


if __name__ == '__main__':

    # Load all the images and GT
    train_imgs, train_class_labels, train_color_labels, test_imgs, test_class_labels, \
        test_color_labels = read_dataset(root_folder='./images/', gt_json='./images/gt.json')

    # List with all the existent classes
    classes = list(set(list(train_class_labels) + list(test_class_labels)))

    # Load extended ground truth
    imgs, class_labels, color_labels, upper, lower, background = read_extended_dataset()
    cropped_images = crop_images(imgs, upper, lower)

    # You can start coding your functions here

    print("===== EXPERIMENT BASE =====")

    kmeans_options = {
        'km_init': 'kmeans++',
        'tolerance': 1,
        'max_iter': 50,
        'fitting': 'WCD'
    }

    knn_options = {
        'feature_type': 'pixels',
        'distance': 'euclidean',
        'voting': 'normal'
    }

    pred_colors, color_percentages = get_color_labels(
        test_imgs,
        K=3,
        options=kmeans_options,
        remove_bg=False
    )

    pred_shapes, shape_confidences = get_shape_labels_with_confidence(
        train_imgs,
        train_class_labels,
        test_imgs,
        k=3,
        options=knn_options
    )

    shape_acc = Get_shape_accuracy(pred_shapes, test_class_labels)
    color_acc = Get_color_accuracy(pred_colors, test_color_labels)

    final_labels = []

    for i in range(len(pred_shapes)):
        label = ""

        colors = pred_colors[i]

        if type(colors) == str:
            colors = [colors]

        for color in colors:
            label += str(color) + " "

        label += str(pred_shapes[i])
        final_labels.append(label)

    print("Accuracy forma base:", shape_acc)
    print("Accuracy color base:", color_acc)
    
    ok_shape = pred_shapes == test_class_labels

    visualize_retrieval(
        test_imgs,
        topN=20,
        info=final_labels,
        ok=ok_shape,
        title='Clasificacion base: color + forma'
    )
    
    red_images = Retrieval_by_color(test_imgs, pred_colors, 'Red', color_percentages)
    print("Imatges trobades amb color Red:", len(red_images))
    
    if len(red_images) > 0:
        visualize_retrieval(
            np.array(red_images),
            topN=min(20, len(red_images)),
            title='Imagenes clasificadas como Red'
        )
        
    shape_query = 'Sandals'
    shape_images = Retrieval_by_shape(test_imgs, pred_shapes, shape_query)
    print("Imatges trobades amb forma", shape_query, ":", len(shape_images))
    
    if len(shape_images) > 0:
        visualize_retrieval(
            np.array(shape_images),
            topN=min(20, len(shape_images)),
            title='Imagenes clasificadas como ' + shape_query
        )


    combined_images = Retrieval_combined(test_imgs, pred_shapes, pred_colors, shape_query, 'Red')
    print("Imatges trobades combinant forma i color:", len(combined_images))
    
    if len(combined_images) > 0:
        visualize_retrieval(
            np.array(combined_images),
            topN=min(20, len(combined_images)),
            title='Imagenes clasificadas como Red ' + shape_query
        )
        
    print("\n===== EXPERIMENT AMB MILLORES PRÒPIES =====")

    kmeans_options2 = {
        'km_init': 'kmeans++',
        'tolerance': 1,
        'max_iter': 50,
        'fitting': 'WCD'
    }

    knn_options2 = {
        'feature_type': 'grid',
        'distance': 'manhattan',
        'voting': 'weighted'
    }

    pred_colors2, color_percentages2 = get_color_labels(
        test_imgs,
        K=3,
        options=kmeans_options2,
        remove_bg=True,
        bg_threshold=40,
        min_percentage=0.08
    )

    pred_shapes2, shape_confidences2 = get_shape_labels_with_confidence(
        train_imgs,
        train_class_labels,
        test_imgs,
        k=5,
        options=knn_options2
    )

    pred_shapes_uncertain = []

    for i in range(len(pred_shapes2)):
        if shape_confidences2[i] < 0.60:
            pred_shapes_uncertain.append('uncertain')
        else:
            pred_shapes_uncertain.append(pred_shapes2[i])

    pred_shapes_uncertain = np.array(pred_shapes_uncertain)

    shape_acc2 = Get_shape_accuracy(pred_shapes2, test_class_labels)
    shape_acc_uncertain = Get_shape_accuracy(pred_shapes_uncertain, test_class_labels)
    color_acc2 = Get_color_accuracy(pred_colors2, test_color_labels)

    final_labels2 = []

    for i in range(len(pred_shapes_uncertain)):
        label = ""

        colors = pred_colors2[i]

        if type(colors) == str:
            colors = [colors]

        for color in colors:
            label += str(color) + " "

        label += str(pred_shapes_uncertain[i])
        final_labels2.append(label)

    print("Accuracy forma millorada:", shape_acc2)
    print("Accuracy forma amb uncertain:", shape_acc_uncertain)
    print("Accuracy color millorada:", color_acc2)

    ok_shape2 = pred_shapes2 == test_class_labels

    visualize_retrieval(
        test_imgs,
        topN=20,
        info=final_labels2,
        ok=ok_shape2,
        title='Clasificacion mejorada: color + forma'
    )

    red_images2 = Retrieval_by_color(test_imgs, pred_colors2, 'Red', color_percentages2)
    print("Imatges trobades amb color Red millorades:", len(red_images2))

    if len(red_images2) > 0:
        visualize_retrieval(
            np.array(red_images2),
            topN=min(20, len(red_images2)),
            title='Imagenes mejoradas clasificadas como Red'
        )
        
    shape_images2 = Retrieval_by_shape(test_imgs, pred_shapes_uncertain, shape_query)
    print("Imatges trobades amb forma millorada", shape_query, ":", len(shape_images2))
    
    if len(shape_images2) > 0:
        visualize_retrieval(
            np.array(shape_images2),
            topN=min(20, len(shape_images2)),
            title='Imagenes mejoradas clasificadas como ' + shape_query
        )
        
    combined_images2 = Retrieval_combined(test_imgs, pred_shapes_uncertain, pred_colors2, shape_query, 'Red')
    print("Imatges trobades combinant forma i color millorades:", len(combined_images2))

    if len(combined_images2) > 0:
        visualize_retrieval(
            np.array(combined_images2),
            topN=min(20, len(combined_images2)),
            title='Imagenes mejoradas clasificadas como Red ' + shape_query
        )

    ranked_images, ranked_scores = Retrieval_combined_ranked(
        test_imgs,
        pred_shapes_uncertain,
        pred_colors2,
        shape_query,
        'Red',
        color_percentages=color_percentages2,
        shape_confidences=shape_confidences2
    )

    print("Imatges combinades ordenades per score:", len(ranked_images))

    if len(ranked_scores) > 0:
        print("Millor score combinat:", ranked_scores[0])

    if len(ranked_images) > 0:
        ranked_info = []

        for i in range(len(ranked_images)):
            ranked_info.append("score=" + str(round(ranked_scores[i], 3)))

        visualize_retrieval(
            np.array(ranked_images),
            topN=min(20, len(ranked_images)),
            info=ranked_info,
            title='Imagenes combinadas ordenadas por score'
        )
    print("\n===== COMPARACIÓ KNN =====")

    knn_results = compare_knn_options(
        train_imgs,
        train_class_labels,
        test_imgs,
        test_class_labels
    )

    for result in knn_results:
        print(result)

    print("\n===== COMPARACIÓ KMEANS =====")

    kmeans_results = compare_kmeans_options(
        test_imgs,
        test_color_labels,
        K=3
    )

    for result in kmeans_results:
        print(result)

    if len(cropped_images) > 0:
        print("\n===== ESTADÍSTIQUES KMEANS SOBRE UNA IMATGE RETALLADA =====")
        Kmean_statistics(cropped_images[0], Kmax=6)