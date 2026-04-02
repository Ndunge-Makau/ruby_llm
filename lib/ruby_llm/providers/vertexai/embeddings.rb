# frozen_string_literal: true

module RubyLLM
  module Providers
    class VertexAI
      # Embeddings methods for the Vertex AI implementation
      module Embeddings
        module_function

        def embedding_url(model:)
          "projects/#{@config.vertexai_project_id}/locations/#{@config.vertexai_location}/publishers/google/models/#{model}:predict" # rubocop:disable Layout/LineLength
        end

        def render_embedding_payload(text, model:, dimensions: nil, with: nil, other: nil) # rubocop:disable Lint/UnusedMethodArgument
          is_multimodal = with.present?

          if is_multimodal
            render_multimodal_payload(text:, dimensions:, with:, other:)
          else
            payload = {
              instances: [text].flatten.map { |t| { text: t.to_s } }
            }
            payload[:parameters] = { dimension: dimensions } if dimensions
            payload
          end
        end

        def render_multimodal_payload(text:, dimensions:, with:, other: nil)
          files = categorize_files(with)

          validate_multimodal_inputs(files)

          instance = {}
          instance[:text] = text.to_s if text
          add_image_instance(instance, image: files[:image].first) if files[:image].any?
          add_video_instance(instance, video: files[:video].first) if files[:video].any?

          payload = {
            instances: [instance]
          }
          payload[:parameters] = { dimension: dimensions } if dimensions
          add_other_parameters(payload, other) if other
          payload
        end

        def validate_multimodal_inputs(files)
          raise ArgumentError, 'This model only supports one image at a time.' if files[:image].size > 1
          raise ArgumentError, 'This model only supports one video at a time.' if files[:video].size > 1
        end

        def categorize_files(files)
          result = { image: [], video: [] }

          Array(files).each do |file|
            case detect_file_type(file)
            when :image
              result[:image] << file
            when :video
              result[:video] << file
            else
              raise ArgumentError, "Unsupported file type for file: #{file}"
            end
          end
          result
        end

        def detect_file_type(file)
          filename = if file.respond_to?(:path)
                       file.path
                     elsif file.is_a?(String)
                       file
                     else
                       return :unknown
                     end
          extension = File.extname(filename).downcase

          return :image if %w[.jpg .jpeg .png .gif .bmp].include?(extension)
          return :video if %w[.avi .flv .mkv .mov .mp4 .mpeg .mpg .webm .wmv].include?(extension)

          :unknown
        end

        def add_image_instance(instance, image:)
          return unless image.present?

          require 'base64'
          image_data = if image.respond_to?(:read)
                         image.read
                       elsif image.is_a?(String) && File.exist?(image)
                         File.binread(image)
                       else
                         image
                       end

          instance[:image] = { bytesBase64Encoded: Base64.strict_encode64(image_data) }
        end

        def add_video_instance(instance, video:)
          return unless video.present?

          require 'base64'
          if video.is_a?(String) && video.start_with?('gs://')
            instance[:video] = { gcsUri: video }
            return
          end

          video_data = if video.respond_to?(:read)
                         video.read
                       elsif video.is_a?(String) && File.exist?(video)
                         File.binread(video)
                       else
                         video
                       end
          instance[:video] = { bytesBase64Encoded: Base64.strict_encode64(video_data) }
        end

        def add_other_parameters(payload, other)
          return unless other.respond_to?(:to_hash) && other.any?

          if other.key?(:instances)
            deep_merge(payload, other)
          else
            other.each do |key, value|
              payload[:instances].first[:video][key] = value.is_a?(Hash) ? JSON.generate(value) : value
            end
          end
        end

        def deep_merge(base, override)
          override.each { |key, value| merge_value(base, key, value) }
          base
        end

        def merge_value(base, key, value)
          if base[key].is_a?(Hash) && value.is_a?(Hash)
            deep_merge(base[key], value)
          elsif base[key].is_a?(Array) && value.is_a?(Array)
            merge_arrays(base[key], value)
          else
            base[key] = value
          end
        end

        def merge_arrays(base_arr, override_arr)
          override_arr.each_with_index do |item, i|
            next unless item.is_a?(Hash) && base_arr[i].is_a?(Hash)

            deep_merge(base_arr[i], item.transform_keys(&:to_sym))
          end
        end

        def parse_embedding_response(response, model:, text:)
          predictions = response.body['predictions']

          if multimodal_embedding_response?(predictions)
            vectors = parse_multimodal_embeddings(predictions)
          else
            vectors = predictions&.map { |p| p.dig('embeddings', 'values') }
            vectors = vectors.first if vectors&.length == 1 && !text.is_a?(Array)
          end
          Embedding.new(vectors:, model:, input_tokens: 0)
        end

        def multimodal_embedding_response?(predictions)
          predictions&.dig(0, 'textEmbedding') ||
            predictions&.dig(0, 'imageEmbedding') ||
            predictions&.dig(0, 'videoEmbeddings')
        end

        def parse_multimodal_embeddings(predictions)
          text_embedding = predictions&.dig(0, 'textEmbedding')
          image_embedding = predictions&.dig(0, 'imageEmbedding')
          video_embedding = predictions&.dig(0, 'videoEmbeddings')

          vectors = {}
          vectors[:text] = text_embedding if text_embedding
          vectors[:image] = image_embedding if image_embedding
          vectors[:video] = video_embedding if video_embedding
          vectors
        end
      end
    end
  end
end
